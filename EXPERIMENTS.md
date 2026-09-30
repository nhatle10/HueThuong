# FLAG 2027 – Team Experiment Tracker

Internal log for tracking experimental runs, model iterations, and validation scores.

---

## NOTICE

1. **Never train on `main` directly**:
   - Create a branch for your idea: `git checkout -b exp/<your-hypothesis>` (e.g. `exp/infonce-loss`).
2. **Every run must have a config file**:
   - Duplicate `configs/baseline.yaml` -> `configs/exp01_<description>.yaml`.
   - Never hardcode new hyperparameters directly into Python scripts.
3. **How to run & evaluate**:
   ```bash
   # 1. Train
   python main.py --config configs/expXX.yaml

   # 2. Generate CodaBench submission package
   python evaluate.py --config configs/expXX.yaml --ckpt output/checkpoints/<ckpt_name>.pth.tar
   ```
4. **Record the result in the table below before merging or switching branches**.

---

## Results

> **Goal**: Lower Equal Error Rate (EER %) is better.  
> Target: Beat our reproduced baseline overall score of **33.20%**.

| Run ID | Date | Author | Branch / Commit | Backbones | Loss / Technique | Eng (Std) EER | Bangla (Std) EER | Eng (Gender) EER | Bangla (Gender) EER | Overall EER ↓ | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **EXP-00a** | 2026-09-29 | Paper Ref | `main` | VGGFace + ECAPA | FOP (OPL + CE) | 32.54% | 38.12% | 32.99% | 44.01% | **36.92%** | Official paper reported baseline. |
| **EXP-00b** | 2026-09-30 | Team | `main` | VGGFace + ECAPA | FOP (OPL + CE, lr=1e-4) | - | - | - | - | **33.20%** | **Official CodaBench Submission #953716**. Full convergence baseline. |
| **EXP-01** | *YYYY-MM-DD* | *Name* | `exp/infonce-loss` | VGGFace + ECAPA | Symmetric InfoNCE Loss | - | - | - | - | **-** | First loss improvement experiment. |

