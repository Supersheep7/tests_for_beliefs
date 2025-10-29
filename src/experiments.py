def run_accuracy():
    print(f"Running experiment: accuracy")
    # TODO: run accuracy(modality)
    # TODO: get visualizations(choose layer/layer+head)
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