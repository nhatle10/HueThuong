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
