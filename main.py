import argparse
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import load_config
from src.data import read_train_data, make_local_val_split
from src.evaluation import evaluate_model
from src.losses import OrthogonalProjectionLoss
from src.models import FOP


class RunningAverage:

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def avg(self):
        return self.total / float(self.steps)


def train_epoch(
    face_feats,
    voice_feats,
    labels,
    model,
    optimizer,
    ce_loss,
    opl_loss,
    alpha,
    device,
):
    model.train()
    face_feats = torch.from_numpy(face_feats).float().to(device)
    voice_feats = torch.from_numpy(voice_feats).float().to(device)
    labels = torch.from_numpy(labels).to(device)

    comb, _, _ = model.train_forward(face_feats, voice_feats, labels)
    loss_opl, _, _ = opl_loss(comb[0], labels)
    loss_soft = ce_loss(comb[1], labels)
    loss = loss_soft + alpha * loss_opl

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), loss_opl.item(), loss_soft.item()


def fit(
    tr_faces,
    tr_voices,
    tr_labels,
    val_faces,
    val_voices,
    val_targets,
    n_class,
    cfg,
):
    device = torch.device(
        "cuda" if cfg.model.cuda and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    os.makedirs(cfg.run.save_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)

    n_samples = tr_faces.shape[0]
    alphas = (
        cfg.train.alpha_list
        if isinstance(cfg.train.alpha_list, list)
        else [float(cfg.train.alpha_list)]
    )

    results = []
    for alpha in alphas:
        print(f"\n{'='*50}\nStarting Training with Alpha = {alpha}\n{'='*50}")
        model = FOP(
            cfg.model,
            cfg.data.train.face_dim,
            cfg.data.train.voice_dim,
            n_class,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=getattr(cfg.train, "weight_decay", 1e-4),
        )
        ce_loss = nn.CrossEntropyLoss().to(device)
        opl_loss = OrthogonalProjectionLoss(device).to(device)

        best_eer = float("inf")
        best_epoch = 0
        patience_count = 0
        eer_history = []

        for epoch in range(1, cfg.train.max_num_epoch + 1):
            perm = np.random.permutation(n_samples)
            f_shuf, v_shuf, l_shuf = (
                tr_faces[perm],
                tr_voices[perm],
                tr_labels[perm],
            )

            pbar = tqdm(
                range(0, n_samples, cfg.train.batch_size),
                desc=f"Epoch {epoch}/{cfg.train.max_num_epoch}",
            )
            for i in pbar:
                fb = f_shuf[i : i + cfg.train.batch_size]
                vb = v_shuf[i : i + cfg.train.batch_size]
                lb = l_shuf[i : i + cfg.train.batch_size]
                total_l, opl_l, ce_l = train_epoch(
                    fb,
                    vb,
                    lb,
                    model,
                    optimizer,
                    ce_loss,
                    opl_loss,
                    alpha,
                    device,
                )
                pbar.set_postfix(
                    loss=f"{total_l:.4f}",
                    ce=f"{ce_l:.4f}",
                    opl=f"{opl_l:.4f}",
                )

            # Evaluate on genuine local validation set
            if val_faces is not None and val_targets is not None:
                val_res = evaluate_model(
                    model, val_faces, val_voices, device, y_true=val_targets
                )
                val_eer, val_auc = val_res.eer, val_res.auc
                eer_history.append(val_eer)
                print(
                    f"[Epoch {epoch:03d}] Local Val: EER={val_eer*100:.2f}%, AUC={val_auc:.4f} | "
                    f"10-Fold: EER={val_res.legacy_eer*100:.2f}%, Acc={val_res.legacy_acc*100:.1f}%"
                )

                if val_eer + cfg.train.min_delta < best_eer:
                    best_eer = val_eer
                    best_epoch = epoch
                    patience_count = 0
                    ckpt_path = os.path.join(
                        cfg.run.save_dir,
                        f"{cfg.model.fusion}_{cfg.run.tag}_{alpha:.2f}_best.pth.tar",
                    )
                    torch.save(
                        {
                            "epoch": epoch,
                            "state_dict": model.state_dict(),
                            "eer": best_eer,
                            "n_class": n_class,
                        },
                        ckpt_path,
                    )
                    print(f"  --> Saved new best checkpoint to {ckpt_path}")
                else:
                    patience_count += 1
                    if patience_count >= cfg.train.patience:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break
            else:
                # Full train mode: save periodically
                ckpt_path = os.path.join(
                    cfg.run.save_dir,
                    f"{cfg.model.fusion}_{cfg.run.tag}_{alpha:.2f}_best.pth.tar",
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "n_class": n_class,
                    },
                    ckpt_path,
                )

        results.append((alpha, best_eer, best_epoch))

    print(f"\n{'='*50}\nSUMMARY ({cfg.run.tag})\n{'='*50}")
    print(f"{'alpha':<8}{'best_val_EER':<15}{'best_epoch'}")
    for a, e, ep in results:
        print(f"{a:<8.2f}{e:<15.4f}{ep}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Face-Voice Retrieval Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Set seeds
    torch.manual_seed(cfg.run.seed)
    random.seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)
    if cfg.model.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.run.seed)

    face_train, voice_train, train_labels = read_train_data(cfg)

    n_val_speakers = getattr(cfg.train, "val_speakers", 10)
    (tr_faces, tr_voices, tr_labels, n_class), (val_faces, val_voices, val_targets) = make_local_val_split(
        face_train,
        voice_train,
        train_labels,
        n_val_speakers=n_val_speakers,
        seed=cfg.run.seed,
    )

    if val_faces is not None:
        n_pos = int(np.sum(val_targets == 1))
        n_neg = int(np.sum(val_targets == 0))
        print(
            f"  [Validation] Open-set split: {len(val_targets)} pairs ({n_pos} pos, {n_neg} neg) from {n_val_speakers} held-out speakers"
        )
    else:
        print("  [Validation] None (training on 100% of available samples)")

    fit(
        tr_faces,
        tr_voices,
        tr_labels,
        val_faces,
        val_voices,
        val_targets,
        n_class,
        cfg,
    )

