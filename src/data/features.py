import os
import numpy as np
import pandas as pd
import torch


def read_train_data(cfg):
    """Loads training features (VGGFace + ECAPA) and labels from config paths."""
    face_path = getattr(cfg.data.train, "faces", getattr(cfg.data.train, "face_features", None))
    voice_path = getattr(cfg.data.train, "voices", getattr(cfg.data.train, "voice_features", None))

    print(f"Reading Train Face: {face_path}")
    face_data = pd.read_csv(face_path, header=None).values
    face_train = face_data[:, : cfg.data.train.face_dim]
    labels_train = face_data[:, -1].astype(int)

    print(f"Reading Train Voice: {voice_path}")
    voice_data = pd.read_csv(voice_path, header=None).values
    voice_train = voice_data[:, : cfg.data.train.voice_dim]

    print(f"  Train: face {face_train.shape} | voice {voice_train.shape}")
    return face_train, voice_train, labels_train


def read_eval_data(cfg, track: str = "no_gender", language: str = "Bangla"):
    """
    Loads dev evaluation features for a given track and language.
    track: 'no_gender' or 'gender'
    language: 'English' or 'Bangla'
    """
    dev_root = cfg.data.dev.root
    face_path = os.path.join(
        dev_root, track, "features", f"{language}_test_faces.csv"
    )
    voice_path = os.path.join(
        dev_root, track, "features", f"{language}_test_voices.csv"
    )

    print(f"Reading Eval Face: {face_path}")
    face_test = pd.read_csv(face_path, header=None).values[
        :, : cfg.data.train.face_dim
    ]

    print(f"Reading Eval Voice: {voice_path}")
    voice_test = pd.read_csv(voice_path, header=None).values[
        :, : cfg.data.train.voice_dim
    ]

    face_test = torch.from_numpy(face_test).float()
    voice_test = torch.from_numpy(voice_test).float()
    return face_test, voice_test


def make_local_val_split(
    face_train: np.ndarray,
    voice_train: np.ndarray,
    labels_train: np.ndarray,
    n_val_speakers: int = 10,
    seed: int = 42,
    genders: np.ndarray | None = None,
    val_gender_matched: bool = False
):
    """
    Creates an open-set validation split from the training dataset.
    Holds out `n_val_speakers` to form genuine 1:1 matching and non-matching verification pairs.
    
    Returns:
        (tr_faces, tr_voices, tr_labels, n_tr_classes), (val_faces, val_voices, val_targets)
    """
    sample_gender = genders[labels_train] if genders is not None else None

    if n_val_speakers <= 0:
        n_classes = int(np.max(labels_train)) + 1
        return (face_train, voice_train, labels_train, n_classes, sample_gender), (None, None, None)

    unique_speakers = np.unique(labels_train)
    rng = np.random.default_rng(seed)
    shuffled_spks = rng.permutation(unique_speakers)

    val_spks = set(shuffled_spks[:n_val_speakers])
    tr_spks = sorted(list(set(shuffled_spks[n_val_speakers:])))
    spk_to_new_id = {s: i for i, s in enumerate(tr_spks)}

    tr_mask = np.isin(labels_train, tr_spks)
    val_mask = np.isin(labels_train, list(val_spks))

    tr_faces = face_train[tr_mask]
    tr_voices = voice_train[tr_mask]
    tr_labels = np.array([spk_to_new_id[l] for l in labels_train[tr_mask]])
    tr_genders = sample_gender[tr_mask] if sample_gender is not None else None

    v_faces = face_train[val_mask]
    v_voices = voice_train[val_mask]
    v_labels = labels_train[val_mask]
    v_gender = sample_gender[val_mask] if sample_gender is not None else None

    # Positive pairs: face and voice from same sample
    pos_faces = v_faces
    pos_voices = v_voices
    pos_targets = np.ones(len(pos_faces), dtype=int)

    # Negative pairs: mismatched face and voice from different speakers
    neg_voices = []
    for i, l in enumerate(v_labels):
        cand = v_labels != l
        if val_gender_matched and v_gender is not None:
            matched = cand & (v_gender == v_gender[i])
            if matched.any():
                cand = matched
        chosen = rng.choice(np.where(cand)[0])
        neg_voices.append(v_voices[chosen])

    neg_voices = np.array(neg_voices)
    neg_targets = np.zeros(len(v_faces), dtype=int)

    val_f = np.vstack([pos_faces, v_faces])
    val_v = np.vstack([pos_voices, neg_voices])
    val_targets = np.concatenate([pos_targets, neg_targets])

    val_f = torch.from_numpy(val_f).float()
    val_v = torch.from_numpy(val_v).float()

    return (tr_faces, tr_voices, tr_labels, len(tr_spks), tr_genders), (val_f, val_v, val_targets)

# Only needed for gender batching / gender-matched val (EXP-04b).
def read_class_genders(meta_csv: str, n_classes: int, train_list: str | None = None) -> np.ndarray:
    """
    Gender per class id (0..n-1): 0 = female, 1 = male.
    Class ids are NOT meta row order: they come from the train list (field 5 = speaker id,
    field 6 = class id), so map id -> class through that file. EXP-04 assumed row k <-> class k,
    which mislabels 34/70 speakers.
    """
    if train_list is None:
        train_list = os.path.join(
            os.path.dirname(meta_csv), "train_set_extracted", "train_set", "train_set", "train_English.txt"
        )
    meta = pd.read_csv(meta_csv, header=None)
    gender_of = dict(zip(meta[0], meta[1].str.strip().str.lower() == "m"))
    lst = pd.read_csv(train_list, sep=r"\s+", header=None)
    genders = np.full(n_classes, -1)
    for spk, label in zip(lst[4], lst[5]):
        genders[int(label)] = int(gender_of[spk])
    assert (genders >= 0).all(), "some class ids have no gender in the train list"
    return genders
