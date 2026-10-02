import argparse
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import load_config
from src.data import read_train_data, make_local_val_split, read_class_genders
from src.evaluation import evaluate_model
from src.losses import OrthogonalProjectionLoss, PreciseAlignmentLoss
from src.models import FOP
from src.utils.paths import alpha_dir, init_run, write_meta



class RunningAverage:
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def avg(self):
        return self.total / float(self.steps)

def make_batches(n_samples, batch_size, genders=None):
    """Index batches. With genders, each batch is single-gender so every in-batch negative is same-gender."""
    if genders is None:
        perm = np.random.permutation(n_samples)
        return [perm[i : i + batch_size] for i in range(0, n_samples, batch_size)]

    batches, leftovers = [], []

    for g in np.unique(genders):
        idx = np.random.permutation(np.where(genders == g)[0])
        full = len(idx) // batch_size * batch_size
        batches += [idx[i : i + batch_size] for i in range(0, full, batch_size)]
        leftovers.append(idx[full:])

    rest = np.random.permutation(np.concatenate(leftovers))
    if len(rest) > 1:
        batches.append(rest)  # one small mixed batch
    np.random.shuffle(batches)

    return batches


def train_epoch(
    face_feats,
    voice_feats,
    labels,
    model,
    optimizer,
    losses,
    weights,
    device,
):
    model.train()
    face_feats = torch.from_numpy(face_feats).float().to(device)
    voice_feats = torch.from_numpy(voice_feats).float().to(device)
    labels = torch.from_numpy(labels).to(device)

    comb, face_embeds, voice_embeds = model.train_forward(face_feats, voice_feats, labels)

    # [1] Get base losses
    loss_opl, _, _ = losses["opl"](comb[0], labels)
    loss_soft = losses["ce"](comb[1], labels)

    alpha_ce = weights.get("alpha_ce", 1.0)
    alpha_opl = weights.get("alpha_opl", 1.0)
    alpha_align = weights.get("alpha_align", 0.0)

    loss = alpha_ce * loss_soft + alpha_opl * loss_opl
    align_val = 0.0

    # [2] Alignment loss (if present in losses dict and weight > 0)
    if "align" in losses and losses["align"] is not None and alpha_align > 0:
        loss_align = losses["align"](face_embeds, voice_embeds)
        loss = loss + alpha_align * loss_align
        align_val = loss_align.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), loss_opl.item(), loss_soft.item(), align_val


def save_training_plots(history, save_dir, run_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["loss"]) + 1)

    # Subplot 1: Losses
    axes[0].plot(epochs, history["loss"], label="Total Loss", color="black", linewidth=2)
    axes[0].plot(epochs, history["ce"], label="Cross-Entropy", linestyle="--")
    axes[0].plot(epochs, history["opl"], label="OPL", linestyle="--")
    if any(a > 0 for a in history.get("align", [])):
        axes[0].plot(epochs, history["align"], label="Alignment", linestyle="--")
    axes[0].set_title(f"Training Losses ({run_name})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Validation Metrics
    if history.get("val_eer") and len(history["val_eer"]) > 0:
        val_epochs = range(1, len(history["val_eer"]) + 1)
        axes[1].plot(val_epochs, history["val_eer"], label="Val EER (%)", color="crimson", linewidth=2)
        best_idx = int(np.argmin(history["val_eer"]))
        best_val = history["val_eer"][best_idx]
        axes[1].scatter(
            best_idx + 1,
            best_val,
            color="blue",
            s=100,
            zorder=5,
            label=f"Best: {best_val:.2f}% (Epoch {best_idx+1})",
        )
        axes[1].set_title("Open-Set Validation EER")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("EER (%)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f"{run_name}_curves.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n  --> Saved training curves plot to {plot_path}")


def fit(tr_faces, tr_voices, tr_labels, tr_genders, val_faces, val_voices, val_targets, n_class, cfg):
    device = torch.device(
        "cuda" if cfg.model.cuda and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    os.makedirs(cfg.run.out_dir, exist_ok=True)

    n_samples = tr_faces.shape[0]
    alphas = getattr(cfg.train, "alpha_list", None)
    use_gender = getattr(cfg.train, "gender_batches", False) and tr_genders is not None

    if alphas is None:
        alphas = [getattr(cfg.train, "alpha_opl", 1.0)]
    elif not isinstance(alphas, list):
        alphas = [float(alphas)]

    results = []
    for alpha in alphas:
        out_dir = alpha_dir(cfg.run.out_dir, alpha, len(alphas) > 1)
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
        losses = {
            "ce": nn.CrossEntropyLoss().to(device),
            "opl": OrthogonalProjectionLoss(device).to(device),
            "align": (
                PreciseAlignmentLoss(temperature=getattr(cfg.train, "temperature", 0.07)).to(device)
                if getattr(cfg.train, "alpha_align", 0.0) > 0
                else None
            ),
        }

        weights = {
            "alpha_ce": getattr(cfg.train, "alpha_ce", 1.0),
            "alpha_opl": getattr(cfg.train, "alpha_opl", alpha),
            "alpha_align": getattr(cfg.train, "alpha_align", 0.0),
        }

        best_eer = float("inf")
        best_epoch = 0
        patience_count = 0
        eer_history = []
        history = {
            "loss": [],
            "ce": [],
            "opl": [],
            "align": [],
            "val_eer": [],
            "val_auc": [],
        }

        for epoch in range(1, cfg.train.max_num_epoch + 1):
            batches = make_batches(n_samples, cfg.train.batch_size, tr_genders if use_gender else None)
            pbar = tqdm(batches, desc=f"Epoch {epoch}/{cfg.train.max_num_epoch}")
            ep_loss, ep_ce, ep_opl, ep_align, n_batches = 0.0, 0.0, 0.0, 0.0, 0
            
            for idx in pbar:
                fb, vb, lb = tr_faces[idx], tr_voices[idx], tr_labels[idx]
                total_l, opl_l, ce_l, align_l = train_epoch(
                    fb,
                    vb,
                    lb,
                    model,
                    optimizer,
                    losses,
                    weights,
                    device,
                )
                ep_loss += total_l
                ep_ce += ce_l
                ep_opl += opl_l
                ep_align += align_l
                n_batches += 1
                pbar.set_postfix(
                    loss=f"{total_l:.4f}",
                    ce=f"{ce_l:.4f}",
                    opl=f"{opl_l:.4f}",
                    align=f"{align_l:.4f}",
                )

            history["loss"].append(ep_loss / max(1, n_batches))
            history["ce"].append(ep_ce / max(1, n_batches))
            history["opl"].append(ep_opl / max(1, n_batches))
            history["align"].append(ep_align / max(1, n_batches))

            # Evaluate on genuine local validation set
            if val_faces is not None and val_targets is not None:
                val_res = evaluate_model(
                    model, val_faces, val_voices, device, y_true=val_targets
                )
                val_eer, val_auc = val_res.eer, val_res.auc
                eer_history.append(val_eer)
                history["val_eer"].append(val_eer * 100.0)
                history["val_auc"].append(val_auc)
                print(
                    f"[Epoch {epoch:03d}] Local Val: EER={val_eer*100:.2f}%, AUC={val_auc:.4f} | "
                    f"10-Fold: EER={val_res.legacy_eer*100:.2f}%, Acc={val_res.legacy_acc*100:.1f}%"
                )

                if val_eer + cfg.train.min_delta < best_eer:
                    best_eer = val_eer
                    best_epoch = epoch
                    patience_count = 0
                    ckpt_path = os.path.join(out_dir, "best.pth.tar")
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
                ckpt_path = os.path.join(out_dir, "best.pth.tar")
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "n_class": n_class,
                    },
                    ckpt_path,
                )

        save_training_plots(history, out_dir, f"{cfg.run.name}_s{cfg.run.seed}")
        write_meta(
            out_dir,
            seed=cfg.run.seed,
            alpha=alpha,
            best_epoch=best_epoch,
            best_val_eer=None if best_eer == float("inf") else best_eer,
            n_class=n_class,
        )
        results.append((alpha, best_eer, best_epoch))

    print(f"\n{'='*50}\nSUMMARY ({cfg.run.tag})\n{'='*50}")
    print(f"{'alpha':<8}{'best_val_EER':<15}{'best_epoch'}")
    for a, e, ep in results:
        print(f"{a:<8.2f}{e:<15.4f}{ep}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Face-Voice Retrieval Model")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override model-init seed (ensemble member). The local val split "
        "stays fixed on run.split_seed (default: run.seed in the config).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Val split seed is decoupled from init seed so ensemble members share one split
    split_seed = getattr(cfg.run, "split_seed", cfg.run.seed)
    if args.seed is not None:
        cfg.run.seed = args.seed

    cfg.run.out_dir = init_run(cfg, args.config)
    print(f"Run directory: {cfg.run.out_dir}")

    torch.manual_seed(cfg.run.seed)
    random.seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)
    if cfg.model.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.run.seed)


    face_train, voice_train, train_labels = read_train_data(cfg)

    n_val_speakers = getattr(cfg.train, "val_speakers", 10)
    genders = read_class_genders(cfg.data.train.meta_csv, int(train_labels.max()) + 1)
    (tr_faces, tr_voices, tr_labels, n_class, tr_genders), (val_faces, val_voices, val_targets) = make_local_val_split(
        face_train, voice_train, train_labels,
        n_val_speakers=n_val_speakers,
        seed=split_seed,
        genders=genders,
        val_gender_matched=getattr(cfg.train, "val_gender_matched", False),
    )

    if val_faces is not None:
        n_pos = int(np.sum(val_targets == 1))
        n_neg = int(np.sum(val_targets == 0))
        print(
            f"  [Validation] Open-set split: {len(val_targets)} pairs ({n_pos} pos, {n_neg} neg) from {n_val_speakers} held-out speakers"
        )
    else:
        print("  [Validation] None (training on 100% of available samples)")

    fit(tr_faces, tr_voices, tr_labels, tr_genders, val_faces, val_voices, val_targets, n_class, cfg)


