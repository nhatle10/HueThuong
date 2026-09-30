import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
import torch


def make_same_label_list(num_samples: int):
    """Dev set pairs alternate: even index = positive, odd index = negative."""
    return [idx % 2 == 0 for idx in range(num_samples)]


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


def calculate_roc(
    thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10
):
    assert embeddings1.shape[0] == embeddings2.shape[0]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
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


def evaluate_pairs(embeddings, actual_issame, nrof_folds=10):
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(
        thresholds,
        embeddings1,
        embeddings2,
        np.asarray(actual_issame),
        nrof_folds=nrof_folds,
    )
    return tpr, fpr, accuracy


def evaluate_model(model, face_test, voice_test, device):
    """Runs inference on face and voice test pairs and returns (EER, AUC)."""
    model.eval()
    face_test = face_test.to(device)
    voice_test = voice_test.to(device)

    with torch.no_grad():
        _, face, voice = model(face_test, voice_test)
        face = face.cpu().detach().numpy()
        voice = voice.cpu().detach().numpy()

        feat_list = []
        for idx in range(len(face)):
            feat_list.append(voice[idx])
            feat_list.append(face[idx])

        issame_lst = make_same_label_list(len(feat_list))
        feat_list = np.asarray(feat_list)

        tpr, fpr, _ = evaluate_pairs(feat_list, issame_lst, nrof_folds=10)
        auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))

    return eer, auc
