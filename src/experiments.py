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
from collections import defaultdict
from utils.funcs import nested_dict
import json
cfg = load_cfg()

def to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: to_dict(v) for k, v in d.items()}
    return d

def asker_kde(model_name=cfg["common"]["model"]):
    zoom_strength = float(input("Enter the zoom strength: "))        
    offset = float(input("Enter the offset: "))
    kernel = input("Enable kernel density? [y/n]: ").strip().lower() == 'y'
    scatter = input("Enable scatter? [y/n]: ").strip().lower() == 'y'
    pca_mod = input("Use PCA instead of probe directions? [y/n]: ").strip().lower() == 'y'
    return zoom_strength, offset, kernel, scatter, pca_mod    

def run_visualizations(model_name=cfg["common"]["model"]):
    print(f"Running visualizations. Extracting projections.")
    model = get_model(model_name=model_name)
    data = get_data()
    print("Residual....")
    activations, labels = get_activations(model, data, 'residual', model_name=model_name)
    full_residual = []
    for layer, acts in activations.items():
        pca_df, probe_df = plot_kde_scatter(data=acts, labels=labels, model=model)
        full_residual.append((pca_df, probe_df))
    print("Heads...")
    activations, labels = get_activations(model, data, 'heads', model_name=model_name)
    heads = [decompose_mha(x) for x in activations.values()]
    print("Debug", heads)
    full_heads = []
    for layer, layer_heads in enumerate(heads):
        row = []
        for head, head_data in enumerate(layer_heads):
            pca_df, probe_df = plot_kde_scatter(data=head_data, labels=labels, model=model)
            row.append((pca_df, probe_df))
        full_heads.append(row)
        
    save_results(full_residual, 'viz', model=model_name) 
    save_results(full_heads, 'viz', model=model_name, modality='heads') 

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
    model_family = model_name.split('_')[0].split('-')[0]
    print("model family:", model_family)
    model = get_model(model_name=model_name)
    control = input("Do you want to use control directions, too? [y/n]: ").strip().lower() == 'y'
    while True:
        modality = input("Choose the target ['residual', 'heads']: ").strip().lower() 
        if modality == 'residual':
            directions = t.load(Path(ROOT / "results" / model_name / cfg["probe"]["probe_type"] / "directions_residual"), weights_only=False)
            if control:
                orth_dirs = []
                for v in directions:
                    r = t.randn_like(v)

                    # orthogonalize r w.r.t. v
                    v_norm_sq = (v * v).sum()
                    r_orth = r - (r * v).sum() / v_norm_sq * v

                    # norm-match
                    r_orth = r_orth / r_orth.norm() * v.norm()

                    orth_dirs.append(r)

                directions_control = t.stack(orth_dirs)
                print(f"Using orthogonal random control directions for {modality}.")
            accuracies = t.load(Path(ROOT / "results" / model_name / cfg["probe"]["probe_type"] / "accuracies_residual"), weights_only=False)
            break
        elif modality == 'heads':
            directions = t.load(Path(ROOT / "results" / model_name / cfg["probe"]["probe_type"] / "directions_heads"), weights_only=False)
            directions = t.stack([t.stack(row) for row in directions])
            resid_mid_directions = t.load(Path(ROOT / "results" / model_name / cfg["probe"]["probe_type"] / "directions_mid"), weights_only=False)
            directions = compute_attention_sign_mask(model, directions, resid_mid_directions)       # Sign the directions based on residual mid directions
            if control:
                orth_dirs = []
                for v in directions:
                    r = t.randn_like(v)

                    # orthogonalize
                    v_norm_sq = (v * v).sum()
                    r_orth = r - (r * v).sum() / v_norm_sq * v

                    # norm match
                    r_orth = r_orth / r_orth.norm() * v.norm()

                    orth_dirs.append(r)

                directions_control = t.stack(orth_dirs)
                print(f"Using orthogonal random control directions for {modality}.")
            accuracies = t.tensor(t.load(Path(ROOT / "results" / model_name / cfg["probe"]["probe_type"] / "accuracies_heads"), weights_only=False))
            break
        else:
            print("Invalid modality. Please choose 'residual' or 'heads'.")
    
    sweep = input("Do you want to run an intervention sweep? [y/n]: ").strip().lower() == 'y'
    if not sweep:
        full_results = nested_dict()
    if sweep:
        while True:
            full_results = nested_dict()
            save_results(full_results, "intervention_scores", model=model_name, modality=modality, notes="debug")
            print("Saved results for debugging")
            x_true, y_true, x_false, y_false = get_data('intervention', sweep=True)
            print("Running intervention sweep...")
            alphas = input("Enter alpha values separated by commas (e.g., 1,3,5): ")
            ks = input("Enter k values separated by commas (e.g., 5,10,20): ")
            alpha_list = [float(a.strip()) for a in alphas.split(',')]
            print(alpha_list)
            alpha_list_flipped = [-a for a in alpha_list]
            k_list = [int(k.strip()) for k in ks.split(',')]
            full_results['sweep']['alphas'] = alpha_list
            full_results['sweep']['ks'] = k_list
            # Falses
            print("False --> True...")
            boolp_f, probdiff_f = parameter_sweep(model_baseline=model, prompts=x_false, activation_accuracies=accuracies, activation_directions=directions, ks=k_list, alphas=alpha_list, labels=y_false, attn=modality=='heads', model_family=model_family)
            print(boolp_f)
            print(probdiff_f)
            full_results['sweep']['ft']['clean'] = (boolp_f, probdiff_f)
            if control: 
                input("Press Enter to continue on the control directions...")
                print("Control False --> True...")
                boolp_f_control, probdiff_f_control = parameter_sweep(model_baseline=model, prompts=x_false, activation_accuracies=accuracies, activation_directions=directions_control, ks=k_list, alphas=alpha_list, labels=y_false, attn=modality=='heads', model_family=model_family)
                print(boolp_f_control)
                print(probdiff_f_control)
                full_results['sweep']['ft']['control'] = (boolp_f_control, probdiff_f_control)
            input("Press Enter to continue on the next direction...")
            # Trues
            print("True --> False...")
            boolp_t, probdiff_t = parameter_sweep(model_baseline=model, prompts=x_true, activation_accuracies=accuracies, activation_directions=directions, ks=k_list, alphas=alpha_list_flipped, labels=y_true, attn=modality=='heads', model_family=model_family)
            print(boolp_t)
            print(probdiff_t)
            full_results['sweep']['tf']['clean'] = (boolp_t, probdiff_t)
            if control: 
                input("Press Enter to continue on the control directions...")
                print("Control True --> False...")
                boolp_t_control, probdiff_t_control = parameter_sweep(model_baseline=model, prompts=x_true, activation_accuracies=accuracies, activation_directions=directions_control, ks=k_list, alphas=alpha_list_flipped, labels=y_true, attn=modality=='heads', model_family=model_family)
                print(boolp_t_control)
                print(probdiff_t_control)
                full_results['sweep']['tf']['control'] = (boolp_t_control, probdiff_t_control)
            retry = input("Do you want to run another sweep? [y/n]: ").strip().lower()
            if retry != 'y':
                break
    x_true, y_true, x_false, y_false = get_data('intervention')
    alpha_list = [0, float(input("Enter alpha value for False --> True: "))]
    print(alpha_list)
    k_list = [int(input("Enter k value for False --> True: "))]
    print(k_list)
    alpha_list_flipped = [0, -float(input("Enter alpha value (absolute) for True --> False: "))]
    print(alpha_list_flipped)
    k_list_flipped = [int(input("Enter k value for True --> False: "))]
    print(k_list_flipped)
    alpha_list_control = [0, float(input("Enter CONTROL alpha value for False --> True: "))]
    print(alpha_list_control)
    k_list_control = [int(input("Enter CONTROL k value for False --> True: "))]
    print(k_list_control)
    alpha_list_control_flipped = [0, -float(input("Enter CONTROL alpha value (absolute) for True --> False: "))]
    print(alpha_list_control_flipped)
    k_list_control_flipped = [int(input("Enter CONTROL k value for True --> False: "))]
    print(k_list_control_flipped)

    full_results['fixed']['ft']['alpha'] = alpha_list
    full_results['fixed']['ft']['k'] = k_list
    full_results['fixed']['tf']['alpha'] = alpha_list_flipped
    full_results['fixed']['tf']['k'] = k_list_flipped
    full_results['fixed']['ft']['alpha_control'] = alpha_list_control
    full_results['fixed']['ft']['k_control'] = k_list_control
    full_results['fixed']['tf']['alpha_control'] = alpha_list_control_flipped
    full_results['fixed']['tf']['k_control'] = k_list_control_flipped

    print("False --> True...")
    boolp_f, probdiff_f = parameter_sweep(model_baseline=model, prompts=x_false, activation_accuracies=accuracies, activation_directions=directions, ks=k_list, alphas=alpha_list, labels=y_false, attn=modality=='heads', model_family=model_family)
    print(boolp_f)
    print(probdiff_f)
    full_results['fixed']['ft']['clean'] = (boolp_f, probdiff_f)
    if control:
        input("Press Enter to continue on the CONTROL directions...")
        print("Control False --> True...")
        boolp_f_control, probdiff_f_control = parameter_sweep(model_baseline=model, prompts=x_false, activation_accuracies=accuracies, activation_directions=directions_control, ks=k_list_control, alphas=alpha_list_control, labels=y_false, attn=modality=='heads', model_family=model_family)
        print(boolp_f_control)
        print(probdiff_f_control)
        full_results['fixed']['ft']['control'] = (boolp_f_control, probdiff_f_control)
    
    input("Press Enter to continue on the next direction...")

    print("True --> False...")
    boolp_t, probdiff_t = parameter_sweep(model_baseline=model, prompts=x_true, activation_accuracies=accuracies, activation_directions=directions, ks=k_list_flipped, alphas=alpha_list_flipped, labels=y_true, attn=modality=='heads', model_family=model_family)
    print(boolp_t)
    print(probdiff_t)

    full_results['fixed']['tf']['clean'] = (boolp_t, probdiff_t)

    if control: 
        input("Press Enter to continue on the CONTROL directions...")
        print("Control True --> False...")
        boolp_t_control, probdiff_t_control = parameter_sweep(model_baseline=model, prompts=x_true, activation_accuracies=accuracies, activation_directions=directions_control, ks=k_list_control_flipped, alphas=alpha_list_control_flipped, labels=y_true, attn=modality=='heads', model_family=model_family)
        print(boolp_t_control)
        print(probdiff_t_control)
        full_results['fixed']['tf']['control'] = (boolp_t_control, probdiff_t_control)

    save_results(boolp_t, "intervention_scores", model=model_name, direction='tf', k=k_list_flipped[-1], alpha=alpha_list_flipped[-1], notes=f"boolp", modality=modality)
    save_results(probdiff_t, "intervention_scores", model=model_name, direction='tf', k=k_list_flipped[-1], alpha=alpha_list_flipped[-1], notes=f"probdiff", modality=modality)
    print(full_results)
    save_results(full_results, "intervention_scores", model=model_name, modality=modality)
    return

def run_coherence(model_name=cfg["common"]["model"]):
    logics = [l.strip() for l in input("Choose the logic(s) (comma-separated) [Possible values: 'neg', 'or', 'and']: ").split(',')]
    estimators = [e.strip() for e in input("Choose the estimator(s) (comma-separated) [Possible values: 'logistic_regression', 'mmp', 'logits', 'self_report']: ").split(',')]
    modality = input("Choose the target ['residual', 'heads']: ").strip().lower()
    if modality == 'residual':
        best_layer = t.load(Path(ROOT / "results" / model_name / cfg["probe"]["probe_type"] / "accuracies_residual"), weights_only=False).index(max(t.load(Path(ROOT / "results" / model_name / cfg["probe"]["probe_type"] / "accuracies_residual"), weights_only=False)))
        print("Loaded best layer:", best_layer)
    elif modality == 'heads':
        accuracies_heads = t.tensor(t.load(Path(ROOT / "results" / model_name / cfg["probe"]["probe_type"] / "accuracies_heads"), weights_only=False))
        best_layer = divmod(accuracies_heads.argmax().item(), accuracies_heads.shape[1])
        print("Loaded best layer and head:", (best_layer[0], best_layer[1]))
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
        estimator = Estimator(estimator_name=e, model=model, best_layer=best_layer, modality=modality)
        print("Loaded best layer:", best_layer)
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
                results, baseline = run_coherence_neg(estimator)
            elif logic == 'or':
                estimator.set_logic('or')
                results = run_coherence_or(estimator)
                baseline = 0.5
            elif logic == 'and':
                estimator.set_logic('and')
                results = run_coherence_and(estimator)
                baseline = 0.5
            elif logic == 'ifthen':
                print("Ifthen experiment was dropped. Skipping...")
                # estimator.set_logic('ifthen')
                # results = run_coherence_ifthen(estimator)
            results_estimator[logic] = (results, baseline)
        results_tot[e] = results_estimator

    print("Coherence experiment completed.")
    print("Results: ", results_tot)

    # print()
    # print("= = = WARNING = = =")
    # print()
    # print("= = = We are still testing, so we won't save the results! = = =")
    # print()
    # print("= = = END WARNING = = =")
    # print()
    save_results(results_tot, "coherence_scores", model=model_name, notes=f"{''.join(logics)}_{'_'.join(estimators)}_{modality}")
    
    return

def run_uniformity(model_name=None):

    # Fetch best layer (we will go with the residual)

    experiment_type = input("Enter experiment type ['logic', 'domain']:")
    modality = input("Choose the target ['residual', 'heads']: ").strip().lower()
    if modality == 'residual':
        best_layer = t.load(Path(ROOT / "results" / model_name / cfg["probe"]["probe_type"] / "accuracies_residual"), weights_only=False).index(max(t.load(Path(ROOT / "results" / model_name / cfg["probe"]["probe_type"] / "accuracies_residual"), weights_only=False)))
        print("Loaded best layer:", best_layer)
    elif modality == 'heads':
        accuracies_heads = t.tensor(t.load(Path(ROOT / "results" / model_name / cfg["probe"]["probe_type"] / "accuracies_heads"), weights_only=False))
        best_layer = divmod(accuracies_heads.argmax().item(), accuracies_heads.shape[1])
        print(accuracies_heads[best_layer[0], best_layer[1]])
        print("Loaded best layer and head:", (best_layer[0], best_layer[1]))
    folds = get_data('uniformity') # folds_logic, folds_domain
    model = get_model(model_name=model_name)
    if experiment_type == 'domain': 
        fold_to_probe = folds[0]
    elif experiment_type == 'logic':
        fold_to_probe = folds[1]
    else: 
        print("Invalid experiment type. Please choose 'logic' or 'domain'.")
        return
    results = results = defaultdict(lambda: defaultdict(list))
    train_datasets = fold_to_probe[0]
    test_datasets = fold_to_probe[1]

    for i, train_set in enumerate(train_datasets):

        print("Openend training set n ", i)
        print("Domains of training set: ", train_set['filename'].unique())
        # train_0, ..., train_n-1
        data = (list(train_set['statement']), list(train_set['label']))
        activations, labels = get_activations(model, data, modality=modality, focus=best_layer, model_name=model_name)
        print(activations.keys())
        activations = next(iter(activations.values()))
        if modality == 'heads':
            heads = decompose_mha(activations)
            activations = heads[best_layer[1]]
        X = einops.rearrange(activations, 'n b d -> (n b) d') # Do we need this? 
        y = einops.rearrange(labels, 'n b -> (n b)')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        probe = SupervisedProbe(X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test,
                        probe_cfg=probe_cfg)
        probe.initialize_probe(override_probe_type='logistic_regression')
        print("Training Probe...")
        probe.train()

        # Normalization loop

        train_mean = X_train.mean(dim=0, keepdim=True)
        train_std = X_train.std(dim=0, keepdim=True, unbiased=False)
        train_std = torch.where(train_std == 0, torch.ones_like(train_std), train_std)
        train_std = torch.nan_to_num(train_std, nan=1.0)

        X_test  = (X_test - train_mean)
        X_test  /= train_std
        y_pred = probe.predict(X_test)
        y_test = y_test.cpu().detach().numpy()
        acc = accuracy_score(y_test, y_pred)
        print("accuracy on first test set: ", acc)
        
        for j, test_set in enumerate(test_datasets):

            # test_0, ... , test_n-1
            print("Domains of test set: ", test_set['filename'].unique())

            data = (list(test_set['statement']), list(test_set['label']))
    
            activations, labels = get_activations(model, data, modality=modality, focus=best_layer, model_name=model_name, batch_size=16)
            print(activations.keys())
            activations = next(iter(activations.values()))
            if modality == 'heads':
                heads = decompose_mha(activations)
                activations = heads[best_layer[1]]
            X = einops.rearrange(activations, 'n b d -> (n b) d')
            y = einops.rearrange(labels, 'n b -> (n b)')

            # Normalization loop

            X = (X - train_mean)
            X /= train_std

            probas = probe.predict(X, proba=True)
            y_pred = probe.predict(X)
            y = y.cpu().detach().numpy()
            acc = accuracy_score(y, y_pred)

            print("accuracy on test set ", j, " : ", acc)

            results[i][j].append(acc)
            print("Running results: ", results)
    results = to_dict(results)
    print("Uniformity experiment completed.")
    print("Results: ", results)              
    save_results(results, f"uniformity", model=model_name, modality='uniformity', notes=f"{experiment_type}"f"{modality}")
    print("Results saved in folder 'ROOT/results'")

    return

EXPERIMENTS = {
    'visualizations': run_visualizations,
    'accuracy': run_accuracy,
    'intervention': run_intervention,
    'coherence': run_coherence,
    'uniformity': run_uniformity
}