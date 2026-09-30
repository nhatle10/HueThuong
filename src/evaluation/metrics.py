import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn import metrics
from sklearn.model_selection import KFold


class EvalResult:
    def __init__(self, eer, auc, legacy_eer, legacy_auc, legacy_acc):
        self.eer = eer
        self.auc = auc
        self.legacy_eer = legacy_eer
        self.legacy_auc = legacy_auc
        self.legacy_acc = legacy_acc

    def __iter__(self):
        # Enables tuple unpacking: eer, auc = evaluate_model(...)
        yield self.eer
        yield self.auc

    def __getitem__(self, idx):
        return [self.eer, self.auc][idx]

    def __repr__(self):
        return (
            f"EvalResult(Challenge[EER={self.eer*100:.2f}%, AUC={self.auc:.4f}] | "
            f"Legacy[10-Fold EER={self.legacy_eer*100:.2f}%, AUC={self.legacy_auc:.4f}, Acc={self.legacy_acc:.4f}])"
        )


def make_same_label_list(num_samples: int):
    """Dev set pairs alternate: even index = positive, odd index = negative."""
    return np.array([idx % 2 == 0 for idx in range(num_samples)], dtype=int)


# Challenge Official Evaluator (Global ROC Curve via Brent's Method)
def calculate_challenge_eer_auc(y_true, dists):
    """
    Computes exact EER and AUC matching challenge/eval_submission.py.
    dists: Euclidean distances (lower = more similar).
    """
    fpr, tpr, _ = metrics.roc_curve(y_true, -dists)
    auc = metrics.auc(fpr, tpr)
    fnr = 1.0 - tpr
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except Exception:
        diffs = np.abs(fpr - fnr)
        eer = fpr[np.nanargmin(diffs)]
    return eer, auc


# Legacy 10-Fold Cross-Validation Evaluator (Saeed et al. ICASSP 2022)
def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame))
    )
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


from sklearn.model_selection import StratifiedKFold


def calculate_legacy_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert embeddings1.shape[0] == embeddings2.shape[0]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    actual_issame = np.asarray(actual_issame)[:nrof_pairs]

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)[:nrof_pairs]

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros(nrof_folds)

    skf = StratifiedKFold(n_splits=nrof_folds, shuffle=True, random_state=42)

    for fold_idx, (train_set, test_set) in enumerate(skf.split(dist, actual_issame)):
        acc_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set]
            )
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = (
                calculate_accuracy(
                    threshold, dist[test_set], actual_issame[test_set]
                )
            )
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index],
            dist[test_set],
            actual_issame[test_set],
        )

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy



def evaluate_legacy_10fold(voice_embeds, face_embeds, actual_issame, nrof_folds=10):
    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy = calculate_legacy_roc(
        thresholds, voice_embeds, face_embeds, np.asarray(actual_issame), nrof_folds=nrof_folds
    )
    auc = metrics.auc(fpr, tpr)
    fnr = 1.0 - tpr
    abs_diffs = np.abs(fpr - fnr)
    eer = fpr[np.nanargmin(abs_diffs)]
    acc = np.mean(accuracy)
    return eer, auc, acc


# =========================================================================
# Unified Evaluator (Runs both and returns EvalResult)
# =========================================================================
def evaluate_model(model, face_test, voice_test, device, y_true=None):
    """
    Runs inference on face and voice test pairs and computes BOTH:
    1. Challenge EER / AUC (Global ROC curve matching eval_submission.py)
    2. Legacy EER / AUC / Accuracy (10-fold cross validation matching online_evaluation.py)
    If y_true is provided, it uses the real labels. If None, falls back to make_same_label_list.
    """
    model.eval()
    face_test = face_test.to(device)
    voice_test = voice_test.to(device)

    with torch.no_grad():
        _, face, voice = model(face_test, voice_test)
        face_np = face.cpu().detach().numpy()
        voice_np = voice.cpu().detach().numpy()

    dists = np.linalg.norm(face_np - voice_np, axis=1)
    if y_true is None:
        y_true = make_same_label_list(len(dists))

    # 1. Challenge evaluation
    ch_eer, ch_auc = calculate_challenge_eer_auc(y_true, dists)

    # 2. Legacy 10-fold evaluation
    leg_eer, leg_auc, leg_acc = evaluate_legacy_10fold(
        voice_np, face_np, y_true, nrof_folds=10
    )

    return EvalResult(ch_eer, ch_auc, leg_eer, leg_auc, leg_acc)

