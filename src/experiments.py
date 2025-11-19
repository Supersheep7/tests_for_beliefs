import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
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
    model = get_model()
    data = get_data()
    modality = input("Choose the target ['residual', 'heads']: ").strip().lower()
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
    else:
        print("Invalid modality. Please choose 'residual' or 'heads'.")
        return

def run_accuracy():
    print(f"Running experiment: accuracy")
    model = get_model()
    data = get_data()
    modality = input("Choose the target ['residual', 'heads']: ").strip().lower()
    if modality == 'residual':
        activations, labels = get_activations(model, data, 'residual')
        accuracies, directions, probes = probe_sweep(activations[layer], labels)
        print_answer = input("Do you want to print the plot? [y/n]: ").strip().lower()
        if print_answer == 'y':
            plot_line(accuracies, title="Residual Stream Probe Accuracies", label=f"Accuracy for model {cfg['common']['model']}")
        else:
            print("Plotting skipped.")
    elif modality == 'heads':
        activations, labels = get_activations(model, data, 'heads')
        heads = [decompose_mha(x) for x in activations.values()]
        for layer in tqdm(range(len(heads)), desc="Layers"):
            accuracies = []
            directions = []
            probes = []
            acc, dir, pro = probe_sweep(heads[layer], labels)
            accuracies.append(acc)
            directions.append(dir)
            probes.append(pro)
        accuracies = np.array(accuracies)
        print_answer = input("Do you want to print the plot? [y/n]: ").strip().lower()
        if print_answer == 'y':
            plot_heat(accuracies, title="Attention Heads Probe Accuracies (Sorted)", model=cfg['common']['model'], probe=cfg['probe']['probe_type'])
        else:
            print("Plotting skipped.")
    else:
        print("Invalid modality. Please choose 'residual' or 'heads'.")
        return
    
    save_results(accuracies, "accuracies", modality='heads')
    save_results(directions, "directions", modality='heads')
    save_results(probes, "probes", modality='heads')
    print("Results saved in folder 'ROOT/results'")

    return

def run_intervention():

    print(f"Running experiment: accuracy")
    model = get_model()
    modality = input("Choose the target ['residual', 'heads']: ").strip().lower()
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
            boolp, probdiff = parameter_sweep(model=model, prompts=x_true, accuracies=accuracies, directions=directions, ks=k_list, alphas=alpha_list_flipped, metric='boolp', labels=y_true, attn=modality=='heads')
            plot_sweep(boolp, k_list, alpha_list, title="Boolp: True --> False")
            plot_sweep(probdiff, k_list, alpha_list, title="Boolp: True --> False")
            # Falses
            print("False --> True...")
            boolp, probdiff = parameter_sweep(model=model, prompts=x_false, accuracies=accuracies, directions=directions, ks=k_list, alphas=alpha_list, metric='boolp', labels=y_false, attn=modality=='heads')
            plot_sweep(boolp, k_list, alpha_list, title="Boolp: True --> False")
            plot_sweep(probdiff, k_list, alpha_list, title="Boolp: True --> False")
            retry = input("Do you want to run another sweep? [y/n]: ").strip().lower()
            if retry != 'y':
                break
    x_true, y_true, x_false, y_false = get_data('intervention')
    alpha_list = [0, float(input("Enter alpha value for False --> True: "))]
    alpha_list_flipped = [0, float(input("Enter alpha value for True --> False: "))]
    k_list = [int(input("Enter k value: "))]
    # Trues
    boolp, probdiff = parameter_sweep(model=model, prompts=x_true, accuracies=accuracies, directions=directions, ks=k_list, alphas=alpha_list_flipped, metric='boolp', labels=y_true, attn=modality=='heads')
    save_results(boolp[1]-boolp[0], f"intervention_boolp_true_to_false_k{k_list}_a{alpha_list_flipped[1]}_{cfg["model"]}", modality=modality)
    save_results(probdiff[1]-probdiff[0], f"intervention_probdiff_true_to_false_k{k_list}_a{alpha_list_flipped[1]}_{cfg["model"]}", modality=modality)
    # Falses
    boolp, probdiff = parameter_sweep(model=model, prompts=x_true, accuracies=accuracies, directions=directions, ks=k_list, alphas=alpha_list_flipped, metric='boolp', labels=y_true, attn=modality=='heads')
    save_results(boolp[1]-boolp[0], f"intervention_boolp_false_to_true_k{k_list}_a{alpha_list[1]}_{cfg["model"]}", modality=modality)
    save_results(probdiff[1]-probdiff[0], f"intervention_probdiff_false_to_true_k{k_list}_a{alpha_list[1]}_{cfg["model"]}", modality=modality)

    return

def run_coherence():
    print(f"Running experiment: accuracy")
    model = get_model()
    logic = input("Choose the logic: ").strip().lower()
    estimators = [e.strip() for e in input("Choose the estimator(s) (comma-separated): ").split(',')]
    results_tot = {}
    for e in estimators:
        estimator = Estimator(estimator_name=e, model=model)
        if logic == 'neg':
            results = run_coherence_neg(estimator)
        elif logic == 'or':
            results = run_coherence_or(estimator)
        elif logic == 'and':
            results = run_coherence_and(estimator)
        elif logic == 'ifthen':
            results = run_coherence_ifthen(estimator)
        results_tot[e] = results

    print("Coherence experiment completed.")
    print("Results: ", results_tot)
    save_results(results_tot, f"coherence_{logic}_{'_'.join(estimators)}_{cfg['common']['model']}", modality='coherence')
    
    return
