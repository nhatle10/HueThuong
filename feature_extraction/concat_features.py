#!/usr/bin/env python3
"""
concat_features.py -- combine several *_cat feature folders of the same modality into one.

Each `features_<enc>_cat/` folder (from extract_features.py) holds [original | scaled new block].
This keeps the original block once and appends every source's new block:

    .venv/bin/python feature_extraction/concat_features.py --modality faces \
        --sources clip_l14_cat agegender_vit_cat --out face3_cat
    -> features_face3_cat/ with faces = [VGGFace | CLIP | age-gender ViT], voices symlinked to original

Same row contract and layout as extract_features.py, so a config only needs the new paths/dims.
"""

import argparse
import os

import numpy as np
import pandas as pd

TRACKS = ("no_gender", "gender")
LANGS = ("English", "Bangla")


def read(path):
    return pd.read_csv(path, header=None).values


def write_csv(path, feats, labels=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for i, vec in enumerate(feats):
            cells = [f"{v:.6g}" for v in vec]
            if labels is not None:
                cells.append(str(int(labels[i])))
            f.write(",".join(cells) + "\n")
    print(f"  wrote {path} ({len(feats)} rows, {feats.shape[1]} dims)", flush=True)


def symlink(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)


def combine(base, sources):
    """base: original block; sources: list of [original | new] arrays -> [original | new1 | new2 ...]"""
    k = base.shape[1]
    for s in sources:
        assert s.shape[0] == base.shape[0], "row count mismatch"
        assert np.allclose(s[:, :k], base, atol=1e-4), "source does not start with the original block"
    return np.hstack([base] + [s[:, k:] for s in sources])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=["faces", "voices"])
    ap.add_argument("--sources", nargs="+", required=True, help="feature tags, e.g. clip_l14_cat agegender_vit_cat")
    ap.add_argument("--out", required=True, help="new feature tag, e.g. face3_cat")
    ap.add_argument("--data_root", default="data/FLAG_Grand_Challenge_2027_15_09_2027")
    args = ap.parse_args()
    mod = args.modality
    other = "voices" if mod == "faces" else "faces"

    tr = os.path.join(args.data_root, "train_set_extracted", "train_set")
    base = read(os.path.join(tr, "features", mod, f"train_English_{mod}.csv"))
    labels, base = base[:, -1].astype(int), base[:, :-1]
    srcs = []
    for t in args.sources:
        v = read(os.path.join(tr, f"features_{t}", mod, f"train_English_{mod}.csv"))
        assert (v[:, -1].astype(int) == labels).all(), f"{t}: labels differ"
        srcs.append(v[:, :-1])
    write_csv(os.path.join(tr, f"features_{args.out}", mod, f"train_English_{mod}.csv"), combine(base, srcs), labels)
    symlink(os.path.join(tr, "features", other, f"train_English_{other}.csv"),
            os.path.join(tr, f"features_{args.out}", other, f"train_English_{other}.csv"))

    dev = os.path.join(args.data_root, "dev_set_extracted", "dev_set")
    for track in TRACKS:
        for lang in LANGS:
            tdir = os.path.join(dev, track)
            base = read(os.path.join(tdir, "features", f"{lang}_test_{mod}.csv"))
            srcs = [read(os.path.join(tdir, f"features_{t}", f"{lang}_test_{mod}.csv")) for t in args.sources]
            write_csv(os.path.join(tdir, f"features_{args.out}", f"{lang}_test_{mod}.csv"), combine(base, srcs))
            symlink(os.path.join(tdir, "features", f"{lang}_test_{other}.csv"),
                    os.path.join(tdir, f"features_{args.out}", f"{lang}_test_{other}.csv"))
    print(f"DONE {args.out}")


if __name__ == "__main__":
    main()
