import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from utils.get_model import get_model
from utils.processing import get_data, get_activations
from utils.probe import *
from cfg import load_cfg

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
            pass #TODO: plot_residual_accuracies(accuracies)
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
            pass #TODO: plot_head_accuracies(tot_accuracies_heads)
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