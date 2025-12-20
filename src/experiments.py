import sys
import torch as t
from torch.serialization import safe_globals
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

def asker_kde(model_name=cfg["common"]["model"]):
    zoom_strength = float(input("Enter the zoom strength: "))        
    offset = float(input("Enter the offset: "))
    kernel = input("Enable kernel density? [y/n]: ").strip().lower() == 'y'
    scatter = input("Enable scatter? [y/n]: ").strip().lower() == 'y'
    pca_mod = input("Use PCA instead of probe directions? [y/n]: ").strip().lower() == 'y'
    return zoom_strength, offset, kernel, scatter, pca_mod    

def run_visualizations(model_name=cfg["common"]["model"]):
    print(f"Running visualizations")
    while True:
        modality = input("Choose the target ['residual', 'heads']: ").strip().lower()
        print("Running experiment on ", modality)
        if modality not in ['residual', 'heads']:
            print("Invalid modality. Please choose 'residual' or 'heads'.")
        else: 
            break
    model = get_model(model_name=model_name)
    data = get_data()
    if modality == 'residual':
        activations, labels = get_activations(model, data, 'residual', model_name=model_name)
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
        activations, labels = get_activations(model, data, 'heads', model_name=model_name)
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

def run_accuracy(model_name=cfg["common"]["model"]):
    print(f"Running experiment: accuracy")
    while True:
        modality = input("Choose the target ['residual', 'heads', 'mid', 'all']: ").strip().lower()
        print("Your input:", modality)
        if modality not in ['residual', 'heads', 'mid', 'all']:
            print("Invalid modality. Please choose 'residual' or 'heads'.")
        else: 
            break
    model = get_model(model_name=model_name)
    data = get_data()
    top_residual_accuracies = None
    top_mid_accuracies = None
    top_heads_accuracies = None
    if modality == 'residual' or modality == 'all':
        activations, labels = get_activations(model, data, 'residual', model_name=model_name)
        accuracies, directions, probes = probe_sweep(activations.values(), labels)
        print_answer = input("Do you want to print the plot? [y/n]: ").strip().lower()
        if print_answer == 'y':
            plot_line(accuracies, title="Residual Stream Probe Accuracies", label=f"Accuracy for model {cfg['common']['model']}")
        else:
            print("Plotting skipped.")
        top_residuals, top_residual_accuracies = get_top_entries(accuracies, n=5)
        print("Top 5 Residual Positions and their Accuracies:", list(zip(top_residuals, top_residual_accuracies)))
        # print("Directions look like:", directions[0])
        save_results(accuracies, "accuracies", model=model_name, modality='residual')
        save_results(directions, "directions", model=model_name, modality='residual')
        save_results(probes, "probes", model=model_name, modality='residual')
    if modality == 'mid' or modality == 'all':
        activations, labels = get_activations(model, data, 'mid', model_name=model_name)
        accuracies, directions, probes = probe_sweep(activations.values(), labels)
        print_answer = input("Do you want to print the plot? [y/n]: ").strip().lower()
        if print_answer == 'y':
            plot_line(accuracies, title="Residual Stream (Mid) Probe Accuracies", label=f"Accuracy for model {cfg['common']['model']}")
        else:
            print("Plotting skipped.")
        top_residuals, top_residual_accuracies = get_top_entries(accuracies, n=5)
        print("Top 5 Residual (Mid) Positions and their Accuracies:", list(zip(top_residuals, top_residual_accuracies)))
        # print("Directions look like:", directions[0])
        save_results(accuracies, "accuracies", model=model_name, modality='mid')
        save_results(directions, "directions", model=model_name, modality='mid')
        save_results(probes, "probes", model=model_name, modality='mid')
    if modality == 'heads' or modality == 'all':
        activations, labels = get_activations(model, data, 'heads', model_name=model_name)
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
        # print("Directions look like:", directions[0])
        print_answer = input("Do you want to print the plot? [y/n]: ").strip().lower()
        if print_answer == 'y':
            plot_heat(accuracies, title="Attention Heads Probe Accuracies (Sorted)", model=cfg['common']['model'], probe=cfg['probe']['probe_type'])
        else:
            print("Plotting skipped.")
        save_results(accuracies, "accuracies", model=model_name, modality='heads')
        save_results(directions, "directions", model=model_name, modality='heads')
        save_results(probes, "probes", model=model_name, modality='heads')

    print("Results saved in folder 'ROOT/results'")

    return

def run_intervention(model_name=cfg["common"]["model"]):

    model = get_model(model_name=model_name)
    modality = input("Choose the target ['residual', 'heads']: ").strip().lower() 
    if modality == 'residual':
        directions = t.load(Path(ROOT / "results" / model_name / cfg["probe"]["probe_type"] / "directions_residual"), weights_only=False)
        accuracies = t.load(Path(ROOT / "results" / model_name / cfg["probe"]["probe_type"] / "accuracies_residual"), weights_only=False)
    elif modality == 'heads':
        directions = t.load(Path(ROOT / "results" / model_name / cfg["probe"]["probe_type"] / "directions_heads"), weights_only=False)
        directions = t.stack([t.stack(row) for row in directions])
        resid_mid_directions = t.load(Path(ROOT / "results" / model_name / cfg["probe"]["probe_type"] / "directions_mid"), weights_only=False)
        directions = compute_attention_sign_mask(model, directions, resid_mid_directions)       # Sign the directions based on residual mid directions
        accuracies = t.tensor(t.load(Path(ROOT / "results" / model_name / cfg["probe"]["probe_type"] / "accuracies_heads"), weights_only=False))
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
            boolp_t, probdiff_t = parameter_sweep(model_baseline=model, prompts=x_true, activation_accuracies=accuracies, activation_directions=directions, ks=k_list, alphas=alpha_list_flipped, labels=y_true, attn=modality=='heads')
            print(boolp_t)
            print(probdiff_t)
            input("Press Enter to continue on the next direction...")
            # Falses
            print("False --> True...")
            boolp_f, probdiff_f = parameter_sweep(model_baseline=model, prompts=x_false, activation_accuracies=accuracies, activation_directions=directions, ks=k_list, alphas=alpha_list, labels=y_false, attn=modality=='heads')
            print(boolp_f)
            print(probdiff_f)
            retry = input("Do you want to run another sweep? [y/n]: ").strip().lower()
            if retry != 'y':
                saveplot = input("Do you want to save the plots? [y/n]: ").strip().lower()
                if saveplot == 'y':
                    for name, metric in zip(["BoolpTF", "ProbdiffTF", "BoolpFT", "ProbdiffFT"],[boolp_t, probdiff_t, boolp_f, probdiff_f]):
                        plot_sweep(metric, k_list, alpha_list, title=name) 
                save_results(boolp_t, "intervention_sweep", model=model_name, direction='tf', notes=f"boolp_ks_{ks}_alphas_{alphas}", modality=modality)
                save_results(boolp_f, "intervention_sweep", model=model_name, direction='ft', notes=f"boolp_ks_{ks}_alphas_{alphas}", modality=modality)
                save_results(probdiff_t, "intervention_sweep", model=model_name, direction='tf', notes=f"pdiff_ks_{ks}_alphas_{alphas}", modality=modality)
                save_results(probdiff_f, "intervention_sweep", model=model_name, direction='ft', notes=f"pdiff_ks_{ks}_alphas_{alphas}", modality=modality)   
                break
    x_true, y_true, x_false, y_false = get_data('intervention')
    alpha_list = [0, float(input("Enter alpha value for False --> True: "))]
    alpha_list_flipped = [0, float(input("Enter alpha value for True --> False: "))]
    k_list = [int(input("Enter k value for False --> True: "))]
    k_list_flipped = [int(input("Enter k value for True --> False: "))]
    # Trues
    save_results(boolp, "intervention_scores", model=model_name, direction='tf', k=k_list_flipped[-1], alpha=alpha_list_flipped[-1], notes=f"boolp", modality=modality)
    save_results(probdiff, "intervention_scores", model=model_name, direction='tf', k=k_list_flipped[-1], alpha=alpha_list_flipped[-1], notes=f"probdiff", modality=modality)
    
    save_results(boolp[1]-boolp[0], f"intervention_boolp_true_to_false_k{k_list_flipped}_a{alpha_list_flipped[1]}_{model_name}", modality=modality)
    save_results(probdiff[1]-probdiff[0], f"intervention_probdiff_true_to_false_k{k_list_flipped}_a{alpha_list_flipped[1]}_{model_name}", modality=modality)
    # Falses
    boolp, probdiff = parameter_sweep(model_baseline=model, prompts=x_true, activation_accuracies=accuracies, activation_directions=directions, ks=k_list, alphas=alpha_list_flipped, labels=y_true, attn=modality=='heads')
    save_results(boolp, "intervention_scores", model=model_name, direction='ft', k=k_list[-1], alpha=alpha_list[-1], notes=f"boolp", modality=modality)
    save_results(probdiff, "intervention_scores", model=model_name, direction='ft', k=k_list[-1], alpha=alpha_list[-1], notes=f"probdiff", modality=modality)

    return

def run_coherence(model_name=cfg["common"]["model"]):
    best_layer = t.load(Path(ROOT / "results" / model_name / cfg["probe"]["probe_type"] / "accuracies_residual"), weights_only=False).index(max(t.load(Path(ROOT / "results" / model_name / cfg["probe"]["probe_type"] / "accuracies_residual"), weights_only=False)))
    print("Loaded best layer:", best_layer)
    logics = [l.strip() for l in input("Choose the logic(s) (comma-separated) [Possible values: 'neg', 'or', 'and']: ").split(',')]
    estimators = [e.strip() for e in input("Choose the estimator(s) (comma-separated) [Possible values: 'logistic_regression', 'mmp', 'logits', 'self_report']: ").split(',')]
    results_tot = {}
    model = get_model(model_name=model_name)
    shots_self = [
                        'Statement: Paris is the capital of France.\nP(True): 0.95',
                        'Statement: The largest bear in the world is currently in Italy.\nP(True): 0.25',
                        'Statement: Milan is the capital of Italy.\nP(True): 0.05',
                        'Statement: Humans have five senses.\nP(True): 0.65',
                ]
    shots_block = "".join("\n\n" + shot for shot in shots_self)
    for e in estimators:
        results_estimator = {}
        estimator = Estimator(estimator_name=e, model=model, best_layer=best_layer)
        estimator.set_context(
                context = f"The sky is blue. This statement is: True \n\nThe earth is flat. This statement is: False \n\n",
                context_self = (
    "I am a fact-checking AI. For each statement, I rate the probability "
    "that the statement is true on a scale from 0 to 1."
    f"{shots_block}\n\n"
)
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
                print("Ifthen experiment was dropped. Skipping...")
                # estimator.set_logic('ifthen')
                # results = run_coherence_ifthen(estimator)
            results_estimator[logic] = results
        results_tot[e] = results_estimator

    print("Coherence experiment completed.")
    print("Results: ", results_tot)

    print()
    print("= = = WARNING = = =")
    print()
    print("= = = We are still testing, so we won't save the results! = = =")
    print()
    print("= = = END WARNING = = =")
    print()
    # save_results(results_tot, "coherence_scores", model=model_name, notes=f"{''.join(logics)}_{'_'.join(estimators)}")
    
    return

def run_uniformity(model_name=None):

    # Fetch best layer (we will go with the residual)

    best_layer = t.load(Path(ROOT / "results" / model_name / cfg["probe"]["probe_type"] / "accuracies_residual"), weights_only=False).index(max(t.load(Path(ROOT / "results" / model_name / cfg["probe"]["probe_type"] / "accuracies_residual"), weights_only=False)))
    print("Loaded best layer:", best_layer)
    model = get_model(model_name=model_name)
    data = get_data('uniformity')
    results = ()

    for fold_n, folds in enumerate(data):

        # folds_logic, folds_domain

        for i, data_sets in enumerate(folds):            
            
            # train_data, test_data
            
            for j, train_set in enumerate(data_sets):

                # train_0, ..., train_n-1
                data = (train_set['statement'], train_set['label'])
                
                activations, labels = get_activations(model, data, 'residual', focus=best_layer, model_name=model_name)
                X = einops.rearrange(activations, 'n b d -> (n b) d') # Do we need this? 
                y = einops.rearrange(labels, 'n b -> (n b)')
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
                probe = SupervisedProbe(X_train=X_train, y_train=y_train,
                                X_test=X_test, y_test=y_test,
                                probe_cfg=probe_cfg)
                probe.initialize_probe(override_probe_type='logistic_regression')
                probe.train()

                for test_set in data_sets[1]:

                    # test_0, ... , test_n-1

                    data = (test_set['statement'], test_set['label'])
            
                    activations, labels = get_activations(model, test_set, 'residual', focus=best_layer, model_name=model_name)
                    X = einops.rearrange(activations, 'n b d -> (n b) d') # Do we need this? 
                    y = einops.rearrange(labels, 'n b -> (n b)')
                    y_pred = probe.predict(X)
                    acc = accuracy_score(y, y_pred)

                    results[fold_n][i][j].append(acc)

    print("Uniformity experiment completed.")
    print("Results: ", results)              
    save_results(results, f"uniformity_{model_name}", modality='uniformity')
    print("Results saved in folder 'ROOT/results'")

    return

EXPERIMENTS = {
    'visualizations': run_visualizations,
    'accuracy': run_accuracy,
    'intervention': run_intervention,
    'coherence': run_coherence,
    'uniformity': run_uniformity
}