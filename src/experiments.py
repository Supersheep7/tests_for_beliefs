import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from sklearn.metrics import accuracy_score
from utils.get_model import get_model
from utils.processing import get_data, get_activations
from utils.viz import plot_line, plot_heat, plot_kde_scatter, plot_sweep
from utils.probe import *
from cfg import load_cfg
from utils.intervention import *
from coherence_experiments import run_coherence_neg, run_coherence_or, run_coherence_and, run_coherence_ifthen
cfg = load_cfg()

def asker_kde():
    zoom_strength = float(input("Enter the zoom strength: "))        
    offset = float(input("Enter the offset: "))
    kernel = input("Enable kernel density? [y/n]: ").strip().lower() == 'y'
    scatter = input("Enable scatter? [y/n]: ").strip().lower() == 'y'
    pca_mod = input("Use PCA instead of probe directions? [y/n]: ").strip().lower() == 'y'
    return zoom_strength, offset, kernel, scatter, pca_mod    

def run_visualizations():
    print(f"Running visualizations")
    while True:
        modality = input("Choose the target ['residual', 'heads']: ").strip().lower()
        print("Running experiment on ", modality)
        if modality not in ['residual', 'heads']:
            print("Invalid modality. Please choose 'residual' or 'heads'.")
        else: 
            break
    model = get_model()
    data = get_data()
    if modality == 'residual':
        activations, labels = get_activations(model, data, 'residual')
        while True:
            layer = int(input("Enter the layer number for visualization: "))
            if 0 <= layer < len(activations):
                break
            print(f"Invalid layer number. Please enter a number between 0 and {len(activations)-1}.")
        while True:
            zoom_strength, offset, kernel, scatter, pca_mod = asker_kde()
            plot_kde_scatter(data=activations[layer], labels=labels, model=model, zoom_strength=zoom_strength, offset=offset, kernel=kernel, scatter=scatter, pca=pca_mod)
            retry = input("Do you want to adjust parameters and re-plot? [y/n]: ").strip().lower()
            if retry != 'y':
                break
    elif modality == 'heads':
        activations, labels = get_activations(model, data, 'heads')
        heads = [decompose_mha(x) for x in activations.values()]
        while True:
            layer = int(input("Enter the layer number for visualization: "))
            head = int(input("Enter the head number for visualization: "))
            if 0 <= layer < len(activations) and 0 <= head < len(activations[layer]):
                break
            print(f"Invalid layer or head number. Please enter valid numbers.")
        while True:
            zoom_strength, offset, kernel, scatter, pca_mod = asker_kde()
            plot_kde_scatter(data=activations[layer][head], labels=labels, model=model, zoom_strength=zoom_strength, offset=offset, kernel=kernel, scatter=scatter, pca=pca_mod)
            retry = input("Do you want to adjust parameters and re-plot? [y/n]: ").strip().lower()
            if retry != 'y':
                break         

def run_accuracy():
    print(f"Running experiment: accuracy")
    while True:
        modality = input("Choose the target ['residual', 'heads']: ").strip().lower()
        print("Your input:", modality)
        if modality not in ['residual', 'heads']:
            print("Invalid modality. Please choose 'residual' or 'heads'.")
        else: 
            break
    model = get_model()
    data = get_data()
    top_residual_accuracies = None
    top_heads_accuracies = None
    if modality == 'residual':
        activations, labels = get_activations(model, data, 'residual')
        accuracies, directions, probes = probe_sweep(activations.values(), labels)
        print_answer = input("Do you want to print the plot? [y/n]: ").strip().lower()
        if print_answer == 'y':
            plot_line(accuracies, title="Residual Stream Probe Accuracies", label=f"Accuracy for model {cfg['common']['model']}")
        else:
            print("Plotting skipped.")
        top_residuals, top_residual_accuracies = get_top_entries(accuracies, n=5)
        print("Top 5 Residual Positions and their Accuracies:", list(zip(top_residuals, top_residual_accuracies)))
    elif modality == 'heads':
        activations, labels = get_activations(model, data, 'heads')
        heads = [decompose_mha(x) for x in activations.values()]
        accuracies = []
        directions = []
        probes = []
        for layer in tqdm(range(len(heads)), desc="Layers"):
            acc, dir, pro = probe_sweep(heads[layer], labels)
            accuracies.append(acc)
            directions.append(dir)
            probes.append(pro)
        accuracies = np.array(accuracies)
        top_heads, top_heads_accuracies = get_top_entries(accuracies, n=5)
        print("Top 5 Heads Positions and their Accuracies:", list(zip(top_heads, top_heads_accuracies)))
        
        print_answer = input("Do you want to print the plot? [y/n]: ").strip().lower()
        if print_answer == 'y':
            plot_heat(accuracies, title="Attention Heads Probe Accuracies (Sorted)", model=cfg['common']['model'], probe=cfg['probe']['probe_type'])
        else:
            print("Plotting skipped.")
    else:
        print("Invalid modality. Please choose 'residual' or 'heads'.")
        return
    
    save_results(accuracies, "accuracies", modality=modality)
    save_results(directions, "directions", modality=modality)
    save_results(probes, "probes", modality=modality)

    print("Results saved in folder 'ROOT/results'")

    return

def run_intervention():

    print(f"Running experiment: accuracy")
    model = get_model()
    modality = input("Choose the target ['residual', 'heads']: ").strip().lower()
    if modality == 'residual':
        directions = Path(ROOT / "results" / cfg["common"]["model"] / cfg["probe"]["probe_type"] / "directions_residual.pkl")
        accuracies = Path(ROOT / "results" / cfg["common"]["model"] / cfg["probe"]["probe_type"] / "accuracies_residual.pkl")
    elif modality == 'heads':
        directions = Path(ROOT / "results" / cfg["common"]["model"] / cfg["probe"]["probe_type"] / "directions_heads.pkl")
        accuracies = Path(ROOT / "results" / cfg["common"]["model"] / cfg["probe"]["probe_type"] / "accuracies_heads.pkl")
    else:
        print("Invalid modality. Please choose 'residual' or 'heads'.")
    sweep = input("Do you want to run an intervention sweep? [y/n]: ").strip().lower() == 'y'
    if sweep:
        while True:
            x_true, y_true, x_false, y_false = get_data('intervention', sweep=True)
            print("Running intervention sweep...")
            alphas = input("Enter alpha values separated by commas (e.g., 1,3,5): ")
            ks = input("Enter k values separated by commas (e.g., 5,10,20): ")
            alpha_list = [float(a.strip()) for a in alphas.split(',')]
            alpha_list_flipped = [-a for a in alpha_list]
            k_list = [int(k.strip()) for k in ks.split(',')]
            # Trues
            print("True --> False...")
            boolp_t, probdiff_t = parameter_sweep(model=model, prompts=x_true, accuracies=accuracies, directions=directions, ks=k_list, alphas=alpha_list_flipped, metric='boolp', labels=y_true, attn=modality=='heads')
            print(boolp_t)
            print(probdiff_t)
            # Falses
            print("False --> True...")
            print(boolp_f)
            print(probdiff_f)
            boolp_f, probdiff_f = parameter_sweep(model=model, prompts=x_false, accuracies=accuracies, directions=directions, ks=k_list, alphas=alpha_list, metric='boolp', labels=y_false, attn=modality=='heads')
            retry = input("Do you want to run another sweep? [y/n]: ").strip().lower()
            if retry != 'y':
                saveplot = input("Do you want to save the plots? [y/n]: ").strip().lower()
                if saveplot == 'y':
                    for metric in [boolp_t, probdiff_t, boolp_f, probdiff_f]:
                        plot_sweep(metric, k_list, alpha_list, title="Boolp: True --> False")    
                break
    x_true, y_true, x_false, y_false = get_data('intervention')
    alpha_list = [0, float(input("Enter alpha value for False --> True: "))]
    alpha_list_flipped = [0, float(input("Enter alpha value for True --> False: "))]
    k_list = [int(input("Enter k value for False --> True: "))]
    k_list_flipped = [int(input("Enter k value for True --> False: "))]
    # Trues
    boolp, probdiff = parameter_sweep(model=model, prompts=x_true, accuracies=accuracies, directions=directions, ks=k_list_flipped, alphas=alpha_list_flipped, metric='boolp', labels=y_true, attn=modality=='heads')
    save_results(boolp[1]-boolp[0], f"intervention_boolp_true_to_false_k{k_list_flipped}_a{alpha_list_flipped[1]}_{cfg['model']}", modality=modality)
    save_results(probdiff[1]-probdiff[0], f"intervention_probdiff_true_to_false_k{k_list_flipped}_a{alpha_list_flipped[1]}_{cfg['model']}", modality=modality)
    # Falses
    boolp, probdiff = parameter_sweep(model=model, prompts=x_true, accuracies=accuracies, directions=directions, ks=k_list, alphas=alpha_list_flipped, metric='boolp', labels=y_true, attn=modality=='heads')
    save_results(boolp[1]-boolp[0], f"intervention_boolp_false_to_true_k{k_list}_a{alpha_list[1]}_{cfg['model']}", modality=modality)
    save_results(probdiff[1]-probdiff[0], f"intervention_probdiff_false_to_true_k{k_list}_a{alpha_list[1]}_{cfg['model']}", modality=modality)

    return

def run_coherence():
    print(f"Running experiment: accuracy")
    model = get_model()
    logics = [l.strip() for l in input("Choose the logic(s) (comma-separated) [Possible values: 'neg', 'or', 'and', 'ifthen']: ").split(',')]
    estimators = [e.strip() for e in input("Choose the estimator(s) (comma-separated) [Possible values: 'logistic_regression', 'mmp', 'logits', 'self_report']: ").split(',')]
    results_tot = {}
    for e in estimators:
        results_estimator = {}
        estimator = Estimator(estimator_name=e, model=model)
        estimator.set_context(
                context = "I am a fact-checking AI. For each statement, I answer whether the statement is true or false.",     
                shots = [
                        'Statement: The Eiffel Tower is located in Berlin.\nAnswer: False',
                        'Statement: Birds can usually fly.\nAnswer: True',
                        'Statement: The sun revolves around the Earth.\nAnswer: False',
                        'Statement: Water boils at 100 degrees Celsius at standard atmospheric pressure.\nAnswer: True',
                ],
                context_self = "I am a fact-checking AI. For each statement, I rate the probability that the statement is true on a scale from 0 to 1.",
                shots_self = [
                        'Statement: Paris is the capital of France.\nP(true): 0.95',
                        'Statement: The largest bear in the world is currently in Italy.\nP(True): 0.25',
                        'Statement: Milan is the capital of Italy.\nP(True): 0.05',
                        'Statement: Humans have five senses.\nP(True): 0.65',
                ]    
            )
        for logic in logics:
            if logic == 'neg':
                estimator.set_logic('neg')
                results = run_coherence_neg(estimator)
            elif logic == 'or':
                estimator.set_logic('or')
                results = run_coherence_or(estimator)
            elif logic == 'and':
                estimator.set_logic('and')
                results = run_coherence_and(estimator)
            elif logic == 'ifthen':
                estimator.set_logic('ifthen')
                results = run_coherence_ifthen(estimator)
            results_estimator[logic] = results
        results_tot[e] = results_estimator

    print("Coherence experiment completed.")
    print("Results: ", results_tot)
    save_results(results_tot, f"coherence_{'_'.join(logics)}_{'_'.join(estimators)}_{cfg['common']['model']}", modality='coherence')
    
    return

def run_uniformity():

    # Fetch best layer (we will go with the residual)

    best_layer = int(input("Enter the layer number for uniformity experiment: "))
    model = get_model()
    data = get_data('uniformity')
    results = ()

    for fold_n, folds in enumerate(data):
        for i, data_sets in enumerate(folds):            
            for j, train_set in enumerate(data_sets[0]):

                activations, labels = get_activations(model, train_set, 'residual', focus=best_layer)
                X = einops.rearrange(activations, 'n b d -> (n b) d') # Do we need this? 
                y = einops.rearrange(labels, 'n b -> (n b)')
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
                probe = SupervisedProbe(X_train=X_train, y_train=y_train,
                                X_test=X_test, y_test=y_test,
                                probe_cfg=probe_cfg)
                probe.initialize_probe(override_probe_type='logistic_regression')
                probe.train()

                for test_set in data_sets[1]:
            
                    activations, labels = get_activations(model, test_set, 'residual', focus=best_layer)
                    X = einops.rearrange(activations, 'n b d -> (n b) d') # Do we need this? 
                    y = einops.rearrange(labels, 'n b -> (n b)')
                    y_pred = probe.predict(X)
                    acc = accuracy_score(y, y_pred)

                    results[fold_n][i][j].append(acc)

    print("Uniformity experiment completed.")
    print("Results: ", results)              
    save_results(results, f"uniformity_{cfg['common']['model']}", modality='uniformity')
    print("Results saved in folder 'ROOT/results'")

    return

EXPERIMENTS = {
    'visualizations': run_visualizations,
    'accuracy': run_accuracy,
    'intervention': run_intervention,
    'coherence': run_coherence,
    'uniformity': run_uniformity
}