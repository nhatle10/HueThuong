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


def evaluate_and_generate_scores(
    cfg, ckpt_path, track="no_gender", heard_lang="English"
):
    unheard_lang = "Bangla" if heard_lang == "English" else "English"
    dev_root = cfg.data.dev.root
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
        dev_root, track, "features", f"{heard_lang}_test_faces.csv"
    )
    voice_h_path = os.path.join(
        dev_root, track, "features", f"{heard_lang}_test_voices.csv"
    )
    face_h, voice_h = read_test_pair_features(
        face_h_path, voice_h_path, face_dim, voice_dim
    )

    face_u_path = os.path.join(
        dev_root, track, "features", f"{unheard_lang}_test_faces.csv"
    )
    voice_u_path = os.path.join(
        dev_root, track, "features", f"{unheard_lang}_test_voices.csv"
    )
    face_u, voice_u = read_test_pair_features(
        face_u_path, voice_u_path, face_dim, voice_dim
    )

    # 2. Build model and load weights
    n_class = 70
    model = FOP(cfg.model, face_dim, voice_dim, n_class).to(device)
    checkpoint = torch.load(
        ckpt_path, weights_only=False, map_location=device
    )
    state_dict = (
        checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    )
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Predict L2 distance scores
    with torch.no_grad():
        _, f_h_emb, v_h_emb = model(face_h.to(device), voice_h.to(device))
        _, f_u_emb, v_u_emb = model(face_u.to(device), voice_u.to(device))

        scores_h = np.linalg.norm(
            f_h_emb.cpu().numpy() - v_h_emb.cpu().numpy(), axis=1
        )
        scores_u = np.linalg.norm(
            f_u_emb.cpu().numpy() - v_u_emb.cpu().numpy(), axis=1
        )

    # 4. Read pair keys
    key_h_path = os.path.join(dev_root, track, f"{heard_lang}_test.txt")
    key_u_path = os.path.join(dev_root, track, f"{unheard_lang}_test.txt")
    with open(key_h_path, "r") as f:
        keys_h = [line.strip().split()[0] for line in f if line.strip()]
    with open(key_u_path, "r") as f:
        keys_u = [line.strip().split()[0] for line in f if line.strip()]

    # 5. Save challenge submission files
    out_dir = os.path.join("output", "sub_score_v4", track)
    os.makedirs(out_dir, exist_ok=True)

    out_h = os.path.join(out_dir, f"sub_score_v4_{heard_lang}_heard.txt")
    out_u = os.path.join(out_dir, f"sub_score_v4_{unheard_lang}_unheard.txt")

    with open(out_h, "w") as f:
        for k, s in zip(keys_h, scores_h):
            f.write(f"{k} {s:.6f}\n")
    with open(out_u, "w") as f:
        for k, s in zip(keys_u, scores_u):
            f.write(f"{k} {s:.6f}\n")

    print(f"  Saved: {out_h} ({len(keys_h)} pairs)")
    print(f"  Saved: {out_u} ({len(keys_u)} pairs)")

    # 6. Compute verification metrics
    eer_h, auc_h = evaluate_model(model, face_h, voice_h, device)
    eer_u, auc_u = evaluate_model(model, face_u, voice_u, device)
    print(f"  -> {heard_lang} (Heard)   : EER = {eer_h*100:.2f}% | AUC = {auc_h:.4f}")
    print(f"  -> {unheard_lang} (Unheard) : EER = {eer_u*100:.2f}% | AUC = {auc_u:.4f}")

    return {
        f"{track}_{heard_lang}_heard": eer_h,
        f"{track}_{unheard_lang}_unheard": eer_u,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to checkpoint (.pth.tar)"
    )
    parser.add_argument(
        "--track",
        type=str,
        default="all",
        choices=["all", "no_gender", "gender"],
    )
    parser.add_argument(
        "--heard_lang",
        type=str,
        default="English",
        choices=["English", "Bangla"],
    )
    parser.add_argument(
        "--create_zip",
        action="store_true",
        default=True,
        help="Create official submission.zip package",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    tracks = ["no_gender", "gender"] if args.track == "all" else [args.track]

    all_metrics = {}
    for t in tracks:
        res = evaluate_and_generate_scores(
            cfg, args.ckpt, track=t, heard_lang=args.heard_lang
        )
        all_metrics.update(res)

    print(f"\n{'='*60}\nEVALUATION SUMMARY ({args.ckpt})\n{'='*60}")
    print(f"{'Condition':<35}{'EER (%)'}")
    print("-" * 50)
    for cond, eer in all_metrics.items():
        print(f"{cond:<35}{eer*100:.2f}%")
    if len(all_metrics) == 4:
        avg_eer = np.mean(list(all_metrics.values())) * 100
        print("-" * 50)
        print(f"{'Overall Average EER':<35}{avg_eer:.2f}%")

    if args.create_zip and len(all_metrics) == 4:
        import zipfile
        zip_path = os.path.join("output", "submission.zip")
        expected_files = [
            "no_gender/sub_score_v4_English_heard.txt",
            "no_gender/sub_score_v4_Bangla_unheard.txt",
            "gender/sub_score_v4_English_heard.txt",
            "gender/sub_score_v4_Bangla_unheard.txt",
        ]
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for rel in expected_files:
                full = os.path.join("output", "sub_score_v4", rel)
                if os.path.exists(full):
                    zf.write(full, arcname=rel)
        print(f"\n[Submission] Successfully packaged {zip_path}")
        print("Ready to upload to the challenge submission portal!")
