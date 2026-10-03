# FLAG 2027 – Team Experiment Tracker

Internal log for tracking experimental runs, model iterations, and validation scores.

---

## NOTICE

1. **Never train on `main` directly**:
   - Create a branch for your idea: `git checkout -b exp/<your-hypothesis>` (e.g. `exp/infonce-loss`).
2. **Every run must have a config file**:
   - Duplicate `configs/baseline.yaml` -> `configs/exp01_<description>.yaml`.
   - Never hardcode new hyperparameters directly into Python scripts.
3. **How to run & evaluate** (everything trains in parallel on the GPU; prefix with `uv run` if you use uv):
   ```bash
   # Ensemble: train seeds 1-4 at once + build submission zip
   scripts/ensemble.sh configs/expXX.yaml
   #   -s "1 2 3 4 5 6"  custom seeds     -j 2  max parallel jobs     -n  train only (no zip)

   # Sweep: many configs x seeds at once, prints a val-EER summary table
   scripts/sweep.sh "configs/exp06_*.yaml"              # seeds 1 2 3, 6 jobs
   #   -s "1 2 3 4"  custom seeds     -j 8  max parallel jobs

   # Single run / single seed
   python main.py --config configs/expXX.yaml --seed 1
   ```
   - Outputs: `output/<run.name>/s<seed>/` (best.pth.tar, config.yaml, meta.json, curves). Submission: `output/<run.name>/submission_s<seeds>/submission.zip`. Logs: `logs/<run.name>/s<seed>.log`.
   - Re-running a command skips seeds that already finished (delete `output/<name>/s<seed>/` to retrain).
   - Compare configs by their **3-seed mean** val EER; single-seed differences under ~2 pts are noise.

   - Extra backbone features: `scripts/extract_all.sh` (encoders in `feature_extraction/extract_features.py`: ecapa2, resnet293, wavlm_large, clip_l14, agegender_vit) writes `features_<enc>/` (replace) and `features_<enc>_cat/` (concat) folders; point a config at them via `data.train.faces/voices`, `face_dim/voice_dim` and `data.dev.features_dir`.

4. **Record the result in the table below before merging or switching branches**.

---

## Results

> **Goal**: Lower Equal Error Rate (EER %) is better.  
> Target: Beat our reproduced baseline overall score of **33.20%**.

| Run ID | Date | Author | Branch / Commit | Backbones | Loss / Technique | Eng (Std) EER | Bangla (Std) EER | Eng (Gender) EER | Bangla (Gender) EER | Overall EER ↓ | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **EXP-00a** | 2026-09-29 | Paper Ref | `main` | VGGFace + ECAPA | FOP (OPL + CE) | 32.54% | 38.12% | 32.99% | 44.01% | **36.92%** | Official paper reported baseline. |
| **EXP-00b** | 2026-09-30 | Team | `main` | VGGFace + ECAPA | FOP (OPL + CE, lr=1e-4) | 28.57% | 28.59% | 33.40% | 42.23% | **33.20%** | **Official Submission #953716**. Full convergence baseline. |
| **EXP-01** | 2026-10-01 | Team | `exp/paeff-alignment` | VGGFace + ECAPA | PAEFF (1-layer, EGFF + Poincaré) | 28.17% | 31.15% | 32.18% | 42.10% | **33.40%** | **Official Submission #955103**. Best local val (16.67%). Beat baseline on 3/4 tracks. |
| **EXP-02** | 2026-10-01 | Team | `exp/paeff-alignment` | VGGFace + ECAPA | PAEFF (2-layer MLP projection) | 32.74% | 35.70% | 41.55% | 45.64% | **38.91%** | **Official Submission #955143**. Failed. 2-layer MLP overfits on 60 speakers (Val EER 22.08%). Revert to 1-layer. |
| **EXP-03** | 2026-10-01 | Team | `exp/paeff-alignment` | VGGFace + ECAPA | PAEFF (1-layer) 4-seed ensemble (seeds 1-4, mean L2 distance) | 27.38% | 29.30% | 31.16% | 41.14% | **32.25%** | **Official Submission #955379**. Config: `exp01_paeff.yaml` + `--seed {1..4}`, fixed val split (split_seed=1). -1.15 vs EXP-01, -0.95 vs EXP-00b. Local val per seed: s1=_, s2=_, s3=_, s4=_. |
| **EXP-04** | 2026-10-02 | Team | `exp/paeff-alignment` | VGGFace + ECAPA | ~~EXP-03 + gender-pure batches + gender-matched val negatives (4-seed ensemble)~~ **INVALID: wrong class->gender mapping** | 25.79% | 31.01% | 35.64% | 44.28% | **34.18%** | **Official Submission #956927**. **INVALID (found 2026-10-02)**: `read_class_genders` assumed class k = meta row k, but class ids come from the train list (id036 -> 0, ...), so 34/70 speakers had the wrong gender. Batches were split into two near-random speaker groups and "matched val" was not gender-matched; this row does not test gender hard negatives, and the matched-val numbers here and in EXP-05/06 are meaningless. Re-test: EXP-04b. Original note: +1.93 vs EXP-03. Gender tracks worse (Eng +4.48, Bangla +3.14) though they were the target. Local val (gender-matched, not comparable to 16.67%): s1=25.83% s2=30.14% s3=28.19% s4=27.22%, best epochs 12-24. Confound: val change also altered checkpoint selection. Suspect same-speaker false negatives in InfoNCE (4-5 per speaker per batch). **Val check (retrained EXP-03 vs EXP-04, 4 seeds each, `exp/eval_val.py`)**: random-neg val 19.24% vs 21.56%; gender-matched val 27.71% vs 27.85%. Matched val shows no gain and does not predict the gender-track drop; per-seed ranking identical to random val, so it adds no information. Decision: keep random-negative val (`val_gender_matched: false`, the default) for epoch selection. Seed spread on val is ~5 pts (16.9-21.7%), so single-seed local val cannot resolve differences under ~2 pts. |
| **EXP-05** | 2026-10-02 | Team | `exp/paeff-alignment` | VGGFace + ECAPA | EXP-03 + face input dropout 0.9 (4 seeds) | 27.98% | 29.45% | 36.46% | 42.37% | **34.06%** | **Official Submission #957010**. CONFOUNDED: seed 4 checkpoint was an unfinished run (epoch 5, val EER 34.58%) averaged into the ensemble. 3-seed val mean 16.48% vs 18.80% for EXP-03 (sweep: dropout 0.5/0.7/0.9 -> 17.96/17.27/16.48%). |
| **EXP-06** | 2026-10-02 | Team | `exp/paeff-alignment` | VGGFace + ECAPA | EXP-03 + face input dropout 0.9 + voice input dropout 0.3 (4 seeds, all healthy) | 27.98% | 29.30% | 34.42% | 41.83% | **33.38%** | **Official Submission #957016**. Worse than EXP-03 (32.25%) by 1.13 despite better local val (3-seed random val 16.25% vs 18.80%, matched val 25.14% vs 27.71%). Face 0.95 and voice 0.5 variants gave no further val gain (16.67%, 16.48%). **Lesson: local val gains from regularization did not transfer to the challenge score** (second time after EXP-04); 10-speaker val is too noisy/optimistic to judge sub-2-pt changes. Calibration done: retrained EXP-03 reproduces 32.25% exactly (see EXP-03r), so this loss is real. |
| **EXP-03r** | 2026-10-02 | Team | `exp/paeff-alignment` | VGGFace + ECAPA | EXP-03 retrained (`exp03_paeff_multiseeds.yaml`, new output layout) | 27.38% | 29.30% | 31.16% | 41.14% | **32.25%** | **Official Submission #957024**. Identical to EXP-03 in every cell: training is deterministic, so reruns reproduce exactly and submission differences are real (only seed choice adds variance). |
| **EXP-07a** | 2026-10-02 | Team | `exp/paeff-alignment` | VGGFace + ECAPA | Ablation: EXP-03 without alignment loss (`alpha_align: 0`, 4 seeds) | 26.59% | 28.31% | 35.44% | 41.69% | **33.01%** | **Official Submission #957034**. +0.76 vs EXP-03. Alignment loss trades tracks: without it standard tracks improve ~1 pt, English gender degrades +4.28. Keep alignment. |
| **EXP-07b** | 2026-10-02 | Team | `exp/paeff-alignment` | VGGFace + ECAPA | Ablation: EXP-03 with gated fusion instead of EGFF (4 seeds) | 27.38% | 32.43% | 31.57% | 42.51% | **33.47%** | **Official Submission #957033**. +1.22 vs EXP-03 (overall computed from cells). Mainly hurts unheard Bangla (no_gender +3.13). Keep EGFF. |
| **EXP-08** | 2026-10-02 | Team | `exp/paeff-alignment` | VGGFace + ECAPA | Mixed ensemble: EXP-03 (4 seeds) + EXP-07a no-align (4 seeds), mean L2 distance | 24.80% | 28.02% | 32.79% | 40.87% | **31.62%** | **Official Submission #957048**. **New best**, -0.63 vs EXP-03. Beats both members on 3/4 cells (Eng std -2.6): members make different errors. `evaluate.py --ckpt` now builds each model from its run's `config.yaml`, so architectures can be mixed. |
| **EXP-04b** | 2026-10-02 | Team | `exp/paeff-alignment` | VGGFace + ECAPA | EXP-03 + gender-pure batches, corrected class->gender mapping; epochs picked on random val (4 seeds) | 32.34% | 40.11% | 31.36% | 39.10% | **35.73%** | **Official Submission #957082**. Works as a gender *specialist*: Bangla gender 39.10% is the best so far (-2.04 vs EXP-03), Eng gender flat (+0.20), but no_gender collapses (Eng +4.96, Bangla +10.81) because the model never sees cross-gender negatives and stops using gender as a cue. Not usable alone. |
| **EXP-09a** | 2026-10-03 | Team | `exp/voice-ecapa2` | VGGFace + ECAPA+ECAPA2 | EXP-03 with voice = [ECAPA, ECAPA2] concat (384-d), 4 seeds | 26.39% | 32.86% | 29.94% | 44.55% | **33.43%** | **Official Submission #958137**. +1.18 vs EXP-03, mostly unheard Bangla (+3.56 std, +3.41 gender); ECAPA2 may carry language-specific cues. Local val 18.16% (EXP-03 19.24%). Not submitted (local val only): ECAPA2 replace 19.20%, ResNet-293 replace 19.13% / concat 19.58%, WavLM-Large concat 18.96%. |
| **EXP-10a** | 2026-10-03 | Team | `exp/voice-ecapa2` | VGGFace+AgeGenderViT + ECAPA | EXP-03 with face = [VGGFace, age-gender ViT CLS] concat (4864-d), 4 seeds | 24.21% | 27.88% | 32.59% | 40.74% | **31.35%** | **Official Submission #958138**. -0.90 vs EXP-03. Local val 17.81%. |
| **EXP-10b** | 2026-10-03 | Team | `exp/voice-ecapa2` | VGGFace+CLIP ViT-L/14 + ECAPA | EXP-03 with face = [VGGFace, CLIP image embedding] concat (4864-d), 4 seeds | 23.41% | 25.04% | 32.59% | 37.60% | **29.66%** | **Official Submission #958159**. **New best**, -2.59 vs EXP-03, -1.21 vs team's previous best (30.87%, #955865). Biggest gains on unheard Bangla (std -4.26, gender -3.54, best ever on both). Local val 17.53% (best of the sweep; order clip < agegender < ecapa2 matched the challenge). |
| **EXP-11a** | 2026-10-03 | Team | `exp/backbone-features` | VGGFace+CLIP-L + ECAPA | Mixed ensemble: EXP-10b (4 seeds) + EXP-10b without alignment loss (4 seeds), mean L2 distance | 24.21% | 24.04% | 35.85% | 38.56% | **30.66%** | **Official Submission #958225**. +1.00 vs EXP-10b alone: unlike EXP-08, the no-align variant adds errors (Eng gender +3.26). Don't mix no-align models on CLIP features. No-align alone: local val 16.77%. |
| **EXP-11b** | 2026-10-03 | Team | `exp/backbone-features` | VGGFace+CLIP-L+AgeGenderViT + ECAPA | EXP-10b with face = [VGGFace, CLIP-L, age-gender ViT] (5632-d), 4 seeds | - | - | - | - | **-** | Not submitted. Local val 18.72% vs 17.53% for CLIP-L alone: age-gender adds nothing once CLIP is present. |
| **EXP-12a** | 2026-10-03 | Team | `exp/backbone-features` | VGGFace+SigLIP2-So400m + ECAPA | EXP-03 with face = [VGGFace, SigLIP 2 So400m/14-224 pooled embedding] (5248-d), 4 seeds | 23.02% | 25.04% | 32.18% | 38.28% | **29.63%** | **Official Submission #958226**. Ties CLIP-L (29.66%). Local val 16.70%. |
| **EXP-12b** | 2026-10-03 | Team | `exp/backbone-features` | VGGFace+OpenCLIP-ViT-H/14 + ECAPA | EXP-03 with face = [VGGFace, OpenCLIP ViT-H/14 LAION-2B projected embedding] (5120-d), 4 seeds | 21.63% | 22.05% | 33.60% | 35.01% | **28.07%** | **Official Submission #958227**. **New best**, -1.59 vs CLIP-L: bigger CLIP helps, mostly on unheard Bangla (std -2.99, gender -2.59; best ever on both). Local val 16.91% (did not rank it above SigLIP 2 / no-align; sub-2-pt local val still unreliable). |
