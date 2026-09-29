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
   .venv/bin/python main.py --config configs/expXX.yaml

   # 2. Generate scores
   .venv/bin/python computeScore.py --config configs/expXX.yaml --ckpt output/checkpoints/<exp_dir>/checkpoint_best.pth.tar --track all
   ```
4. **Record the result in the table below before merging or switching branches**.

---

## Results

> **Goal**: Lower Equal Error Rate (EER %) is better.  
> Target: Beat the official baseline overall score of **36.92%**.

| Run ID | Date | Author | Branch / Commit | Backbones | Loss / Technique | Eng (Std) EER | Bangla (Std) EER | Eng (Gender) EER | Bangla (Gender) EER | Overall EER ↓ | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **EXP-00** | 2026-09-29 | Baseline | `main` | VGGFace + ECAPA | FOP (OPL + CE) | 32.54% | 38.12% | 32.99% | 44.01% | **36.92%** | Official paper baseline. Huge drop on gender-controlled Bangla. |
| **EXP-01** | *YYYY-MM-DD* | *Name* | `main` | XXXX-XXXX | XXXX | - | - | - | - | **-** | First local reproduction run. |
| **EXP-XX** | | | | | | | | | | | |
