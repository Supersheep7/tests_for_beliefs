import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from utils.get_model import get_model
from utils.processing import get_data, get_activations
from utils.viz import plot_line, plot_heat, plot_kde_scatter
from utils.probe import *
from cfg import load_cfg
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
            print("Plotting skipped. Accuracies data is saved in folder 'ROOT/data'")
    elif modality == 'heads':
        activations, labels = get_activations(model, data, 'heads')
        for layer in tqdm(range(len(activations)), desc="Layers"):
            tot_accuracies_heads = []
            tot_directions_heads = []
            tot_probes_heads = []
            accuracies, directions, probes = probe_sweep(activations[layer], labels)
            tot_accuracies_heads.append(accuracies)
            tot_directions_heads.append(directions)
            tot_probes_heads.append(probes)
        tot_accuracies_heads = np.array(tot_accuracies_heads)
        print_answer = input("Do you want to print the plot? [y/n]: ").strip().lower()
        if print_answer == 'y':
            plot_heat(tot_accuracies_heads, title="Attention Heads Probe Accuracies (Sorted)", model=cfg['common']['model'], probe=cfg['probe']['probe_type'])
        else:
            print("Plotting skipped. Accuracies data is saved in folder 'ROOT/data'")
    else:
        print("Invalid modality. Please choose 'residual' or 'heads'.")
        return
    
    return

def run_intervention():
    print(f"Running experiment: accuracy")
    # TODO: function for the replicator to sweep 
    # TODO: get actual results
    return

def run_coherence():
    print(f"Running experiment: accuracy")
    # TODO: run coherence(logic, model, estimator)
    return

def run_self_consistency():
    # TODO: run self consistency(model, estimator)
    return

def run_analysis():
    print(f"Running data analysis")
    return

EXPERIMENTS = {
    "accuracy": run_accuracy,
    "intervention": run_intervention,
    "coherence": run_coherence,
    "self_consistency": run_self_consistency,
    "analysis": run_analysis
}