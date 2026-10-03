import argparse
import os
import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.evaluation import evaluate_model
from src.models import FOP


def read_test_pair_features(face_path, voice_path, face_dim=4096, voice_dim=192):
    face_test = pd.read_csv(face_path, header=None).values[:, :face_dim]
    voice_test = pd.read_csv(voice_path, header=None).values[:, :voice_dim]
    return torch.from_numpy(face_test).float(), torch.from_numpy(voice_test).float()


def load_run_model_cfg(ckpt_path, cfg):
    run_cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")
    if not os.path.exists(run_cfg_path):
        return cfg.model
    run_cfg = load_config(run_cfg_path)
    for key in ("face_dim", "voice_dim"):
        if getattr(run_cfg.data.train, key) != getattr(cfg.data.train, key):
            raise ValueError(
                f"{ckpt_path}: {key}={getattr(run_cfg.data.train, key)} differs from --config "
                f"({getattr(cfg.data.train, key)}); ensemble members must share input features"
            )
    # same dims is not enough (ECAPA and ECAPA2 are both 192-d): the feature folder must match too
    run_feat = getattr(run_cfg.data.dev, "features_dir", "features")
    cli_feat = getattr(cfg.data.dev, "features_dir", "features")
    if run_feat != cli_feat:
        raise ValueError(
            f"{ckpt_path}: trained with dev features '{run_feat}' but --config uses '{cli_feat}'; "
            "ensemble members must share input features"
        )
    return run_cfg.model


def evaluate_and_generate_scores(
    cfg,
    ckpt_paths,
    track="no_gender",
    heard_lang="English",
    compute_dummy_metrics=False,
    out_dir="output/sub_score_v4"
):
    unheard_lang = "Bangla" if heard_lang == "English" else "English"
    dev_root = cfg.data.dev.root
    # sub-folder of each track holding the dev CSVs, e.g. "features_ecapa2_cat" (see extract_features.py)
    features_dir = getattr(cfg.data.dev, "features_dir", "features")
    face_dim = cfg.data.train.face_dim
    voice_dim = cfg.data.train.voice_dim
    device = torch.device(
        "cuda" if cfg.model.cuda and torch.cuda.is_available() else "cpu"
    )

    print(
        f"\n{'='*50}\nEvaluating Track: {track} | Heard: {heard_lang} | Unheard: {unheard_lang}"
    )

    # 1. Load heard and unheard test pairs
    face_h_path = os.path.join(
        dev_root, track, features_dir, f"{heard_lang}_test_faces.csv"
    )
    voice_h_path = os.path.join(
        dev_root, track, features_dir, f"{heard_lang}_test_voices.csv"
    )
    face_h, voice_h = read_test_pair_features(
        face_h_path, voice_h_path, face_dim, voice_dim
    )

    face_u_path = os.path.join(
        dev_root, track, features_dir, f"{unheard_lang}_test_faces.csv"
    )
    voice_u_path = os.path.join(
        dev_root, track, features_dir, f"{unheard_lang}_test_voices.csv"
    )
    face_u, voice_u = read_test_pair_features(
        face_u_path, voice_u_path, face_dim, voice_dim
    )

    if isinstance(ckpt_paths, str):
        ckpt_paths = [ckpt_paths]

    all_h, all_u = [], []
    for ckpt_path in ckpt_paths:
        checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)
        n_class = checkpoint.get("n_class", 70) if isinstance(checkpoint, dict) else 70
        # Build each model from the config saved next to its checkpoint (output/<run>/s<seed>/config.yaml),
        # so ensembles can mix architectures (e.g. egff + gated). Falls back to --config.
        model_cfg = load_run_model_cfg(ckpt_path, cfg)
        model = FOP(model_cfg, face_dim, voice_dim, n_class).to(device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            _, f_h_emb, v_h_emb = model(face_h.to(device), voice_h.to(device))
            _, f_u_emb, v_u_emb = model(face_u.to(device), voice_u.to(device))
            all_h.append(np.linalg.norm(f_h_emb.cpu().numpy() - v_h_emb.cpu().numpy(), axis=1))
            all_u.append(np.linalg.norm(f_u_emb.cpu().numpy() - v_u_emb.cpu().numpy(), axis=1))
    scores_h = np.mean(all_h, axis=0)
    scores_u = np.mean(all_u, axis=0)

    # 4. Read pair keys
    key_h_path = os.path.join(dev_root, track, f"{heard_lang}_test.txt")
    key_u_path = os.path.join(dev_root, track, f"{unheard_lang}_test.txt")
    with open(key_h_path, "r") as f:
        keys_h = [line.strip().split()[0] for line in f if line.strip()]
    with open(key_u_path, "r") as f:
        keys_u = [line.strip().split()[0] for line in f if line.strip()]

    # 5. Save challenge submission files
    out_dir = os.path.join(out_dir, track)
    os.makedirs(out_dir, exist_ok=True)

    out_h = os.path.join(out_dir, f"sub_score_v4_{heard_lang}_heard.txt")
    out_u = os.path.join(out_dir, f"sub_score_v4_{unheard_lang}_unheard.txt")

    with open(out_h, "w") as f:
        for k, s in zip(keys_h, scores_h):
            f.write(f"{k} {s:.6f}\n")
    with open(out_u, "w") as f:
        for k, s in zip(keys_u, scores_u):
            f.write(f"{k} {s:.6f}\n")

    print(f"  [Inference] Saved {heard_lang} (Heard)   : {out_h} ({len(keys_h)} trials)")
    print(f"  [Inference] Saved {unheard_lang} (Unheard) : {out_u} ({len(keys_u)} trials)")

    metrics_res = {}
    if compute_dummy_metrics:
        if len(ckpt_paths) > 1:
            raise ValueError("--compute_dummy_metrics supports a single checkpoint only")
        res_h = evaluate_model(model, face_h, voice_h, device)
        res_u = evaluate_model(model, face_u, voice_u, device)
        print(
            f"    -> [Dummy Metric] {heard_lang}   : EER={res_h.eer*100:.2f}% (AUC={res_h.auc:.4f})"
        )
        print(
            f"    -> [Dummy Metric] {unheard_lang} : EER={res_u.eer*100:.2f}% (AUC={res_u.auc:.4f})"
        )
        metrics_res[f"{track}_{heard_lang}_heard"] = (res_h.eer, res_h.legacy_eer)
        metrics_res[f"{track}_{unheard_lang}_unheard"] = (res_u.eer, res_u.legacy_eer)

    return {
        "trials_h": len(keys_h),
        "trials_u": len(keys_u),
        "metrics": metrics_res,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Challenge Submissions")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--run", type=str, default=None,
                        help="Run directory, e.g. output/exp01_paeff (use with --seeds)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds to ensemble from --run")
    parser.add_argument("--ckpt", type=str, nargs="+", default=None,
                        help="Explicit checkpoint path(s); alternative to --run/--seeds")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Where to write scores and submission.zip")
    parser.add_argument("--track", type=str, default="all",
                        choices=["all", "no_gender", "gender"])
    parser.add_argument("--heard_lang", type=str, default="English",
                        choices=["English", "Bangla"])
    parser.add_argument("--create_zip", action="store_true", default=True)
    parser.add_argument("--compute_dummy_metrics", action="store_true", default=False)
    args = parser.parse_args()

    if args.run and args.seeds:
        ckpts = [os.path.join(args.run, f"s{s}", "best.pth.tar") for s in args.seeds]
        tag = "-".join(str(s) for s in args.seeds)
        out_root = args.out_dir or os.path.join(args.run, f"submission_s{tag}")
    elif args.ckpt:
        ckpts = args.ckpt
        out_root = args.out_dir or os.path.join("output", "sub_score_v4")
    else:
        parser.error("provide either --run with --seeds, or --ckpt")

    missing = [c for c in ckpts if not os.path.exists(c)]
    if missing:
        parser.error(f"checkpoint(s) not found: {missing}")

    if args.run and args.seeds:
        # meta.json is written when training finishes; without it the checkpoint may be a partial run
        unfinished = [c for c in ckpts if not os.path.exists(os.path.join(os.path.dirname(c), "meta.json"))]
        if unfinished:
            parser.error(f"unfinished runs (no meta.json), retrain these seeds: {unfinished}")

    cfg = load_config(args.config)
    tracks = ["no_gender", "gender"] if args.track == "all" else [args.track]

    all_res = {}
    for t in tracks:
        all_res[t] = evaluate_and_generate_scores(
            cfg,
            ckpts,
            track=t,
            heard_lang=args.heard_lang,
            compute_dummy_metrics=args.compute_dummy_metrics,
            out_dir=out_root,
        )

    expected_files = [
        "no_gender/sub_score_v4_English_heard.txt",
        "no_gender/sub_score_v4_Bangla_unheard.txt",
        "gender/sub_score_v4_English_heard.txt",
        "gender/sub_score_v4_Bangla_unheard.txt",
    ]

    print(f"\n{'='*75}\nSUBMISSION SUMMARY ({'+'.join(ckpts)})\n{'='*75}")
    print(f"{'Cell':<45}{'Status':<15}{'Trials'}")
    print("-" * 75)
    for rel in expected_files:
        full = os.path.join(out_root, rel)
        status = "Ready" if os.path.exists(full) else "Missing"
        lines = 0
        if os.path.exists(full):
            with open(full) as f:
                lines = sum(1 for _ in f)
        print(f"{rel:<45}{status:<15}{lines}")
    print("-" * 75)

    if args.create_zip and len(all_res) == 2:
        import zipfile

        zip_path = os.path.join(out_root, "submission.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for rel in expected_files:
                full = os.path.join(out_root, rel)
                if os.path.exists(full):
                    zf.write(full, arcname=rel)
        print(f"\n[Submission Package] Successfully created: {zip_path}")
        print("(Note: Official EER is computed on the challenge server because dev_set labels are held out.)\n")
