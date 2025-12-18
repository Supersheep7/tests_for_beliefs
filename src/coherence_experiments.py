import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from utils.processing import get_data, get_activations
from utils.probe import *
from cfg import load_cfg
from utils.intervention import *
cfg = load_cfg()

def run_coherence_neg(estimator):
    train_data, data_pos, data_neg = get_data(experiment='coherence', logic='neg')

    if estimator.estimator_name in ['logistic_regression', 'mmp']:
        estimator.set_train_data(train_data)
        estimator.train_estimator()

    probas_pos = estimator.extract_proba(data_pos)
    probas_neg = estimator.extract_proba(data_neg)

    pos_neg = probas_pos + probas_neg

    swapped_pos = probas_pos[t.randperm(len(probas_pos))]
    swapped_neg = probas_neg[t.randperm(len(probas_neg))]
    swapped = swapped_pos + swapped_neg

    ceiling = t.ones_like(pos_neg)
    mse = t.mean((pos_neg - ceiling)**2)
    random_baseline = (t.mean((swapped - ceiling)**2))
    score = 1/(1 + mse)
    random_score = 1/(1 + random_baseline)

    print("Mse", mse)
    print("Baseline_Mse", random_baseline)
    print("Score", score)
    print("Baseline_Score", random_score)
    return score

def run_coherence_or(estimator):
    train_data, data_atom, data_or = get_data(experiment='coherence', logic='or')

    if estimator.estimator_name in ['logistic_regression', 'mmp']:
        estimator.set_train_data(train_data)
        estimator.train_estimator()

    probas_atom = estimator.extract_proba(data_atom)
    probas_or = estimator.extract_proba(data_or)

    corrects = (probas_or >= probas_atom).astype(int)
    score = t.mean(corrects)
    return score

def run_coherence_and(estimator):
    train_data, data_atom, data_or = get_data(experiment='coherence', logic='and')

    if estimator.estimator_name in ['logistic_regression', 'mmp']:
        estimator.set_train_data(train_data)
        estimator.train_estimator()

    probas_atom = estimator.extract_proba(data_atom)
    probas_and = estimator.extract_proba(data_or)

    corrects = (probas_and <= probas_atom).astype(int)
    score = t.mean(corrects)
    return score

def run_coherence_ifthen(estimator):
    train_data, data_atom, data_and, data_ifthen = get_data(experiment='coherence', logic='ifthen')

    if estimator.estimator_name in ['logistic_regression', 'mmp']:
        estimator.set_train_data(train_data)
        estimator.train_estimator()

    probas_atom = estimator.extract_proba(data_atom)
    probas_and = estimator.extract_proba(data_and)
    probas_ifthen = estimator.extract_proba(data_ifthen)

    probas_bayesed = probas_and/probas_atom

    score = 1/(1 + t.mean(np.abs(probas_ifthen - probas_bayesed)))

    return score