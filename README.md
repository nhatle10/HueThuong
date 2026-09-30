# FLAG 2027 Grand Challenge  

**Face-voice Association across LAnguages and Gender (FLAG) 2027** Accepted as an ICASSP 2027 Grand Challenge, supported by the IEEE Signal Processing Society Challenge Program.
For details, please follow [evaluation plan](https://arxiv.org/abs/2609.17913) and [Grand Challenge Website](https://mavceleb.github.io/dataset/competition.html)



---
## Registration
The following [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSeJH4uvcWNjqSi7CssNthIv80GdrszjIvuNp4UN77-KMNzgZg/viewform?usp=sharing&ouid=112056064502733681727) will be used to allow participants to register their teams to the challenge.

## Task

Face-voice association is established through a **cross-modal verification** task: given a single sample containing both a face and a voice, verify whether both belong to the same identity.

FLAG 2027 extends this along two dimensions, evaluated as separate tracks:

| Track | Description |
| --- | --- |
| **Language impact** | Whether a voice recorded in a different language is still correctly associated with the speaker's face. Evaluated on a *heard* language (seen in training) and an *unheard* language. |
| **Gender impact** | Gender-controlled verification — negative pairs are restricted to speakers of the same gender as the positive, removing gender as an identity shortcut. |

Train and test splits follow the **unseen–unheard** configuration: disjoint
speakers, disjoint languages.

## Dataset

Built on **MAV-Celeb v4**, extending the prior FAME 2024 and FAME 2026 splits
with a new set of **100 English–Bengali speakers**, annotated for language and
gender. Samples are drawn from YouTube interviews, talk shows, and television
debates, and include real-world variation in pose, lighting, motion blur,
occlusion, background chatter, and compression artifacts.

The [download folder](https://drive.google.com/drive/folders/1YFVLHIWu0yBQYOIgjTvfK7d_M_fsfzn8?usp=sharing) contains the training and development sets, including both the raw data files and the corresponding pre-extracted audio-visual features. The meta file for training set with gender details can be downloaded [here](https://drive.google.com/file/d/1TLaDoCW5Eh18uhPSu-BhEiGxbL6kNwWJ/view?usp=sharing).  

```
train_set/
├── features/
│   ├── faces/
│   │   └── train_English_faces.csv      # 6,485 rows × 4,097 cols (4,096-d face embedding + class label)
│   └── voices/
│       └── train_English_voices.csv     # 6,485 rows × 193 cols (192-d voice embedding + class label)
└── train_set/
    ├── train_English.txt                # 6,485 lines, file list with labels (format below)
    ├── faces/
    │   └── English/
    │       ├── id001/
    │       │   ├── 00036.jpg
    │       │   ├── 00056.jpg
    │       │   └── ...
    │       ├── id002/
    │       └── ... id070/               # 70 identities, 6,485 .jpg total
    └── voices/
        └── English/
            ├── id001/
            │   ├── 00036.wav
            │   ├── 00056.wav
            │   └── ...
            ├── id002/
            └── ... id070/               # 70 identities, 6,485 .wav total
```

```
dev_set/
├── gender/
│   ├── English_test.txt                 # 982 lines
│   ├── Bangla_test.txt                  # 1,468 lines
│   ├── features/
│   │   ├── English_test_faces.csv       # 982 × 4,096
│   │   ├── English_test_voices.csv      # 982 × 192
│   │   ├── Bangla_test_faces.csv        # 1,468 × 4,096
│   │   └── Bangla_test_voices.csv       # 1,468 × 19
│   ├── English_test/
│   │   ├── faces/    00000.jpg ... (982 files)
│   │   └── voices/   00000.wav ... (982 files)
│   └── Bangla_test/
│       ├── faces/    00000.jpg ... (1,468 files)
│       └── voices/   00000.wav ... (1,468 files)
└── no_gender/
    ├── English_test.txt                 # 1,008 line
    ├── Bangla_test.txt                  # 1,406 lines
    ├── features/
    │   ├── English_test_faces.csv       # 1,008 × 4,096
    │   ├── English_test_voices.csv      # 1,008 × 19
    │   ├── Bangla_test_faces.csv        # 1,406 × 4,096
    │   └── Bangla_test_voices.csv       # 1,406 × 19
    ├── English_test/
    │   ├── faces/    00000.jpg ... (1,008 files)
    │   └── voices/   00000.wav ... (1,008 files)
    └── Bangla_test/
        ├── faces/    00000.jpg ... (1,406 files)
        └── voices/   00000.wav ... (1,406 files)
```

Pair list files use the format:

```
ysuvkz41  voices/English/00000.wav  faces/English/00000.jpg
tog3zj45  voices/English/00001.wav  faces/English/00001.jpg
```

## Baseline

The baseline is a two-branch network over pre-extracted face and voice embeddings, trained with an orthogonality constraint on the multimodal embeddings of different speakers. It follows *Fusion and Orthogonal Projection for Improved Face-Voice Association* ([paper](https://ieeexplore.ieee.org/abstract/document/9747704), [code](https://github.com/msaadsaeed/FOP)).


| Component | Model |
| --- | --- |
| Face encoder | VGGFace |
| Voice encoder | ECAPA-TDNN |
| Fusion | Two-branch network with orthogonal projection |

## Results

| Phase | Config.    | Standard Eng. test | Standard Bengali test | Gender-Constrained Eng. test | Gender-Constrained Bengali test | Overall Score |
|-------|------------|-------------------:|----------------------:|-----------------------------:|--------------------------------:|--------------:|
| Dev   | Eng. train | 32.54              | 38.12                 | 32.99                        | 44.01                           | 36.92         |
| Eval  | Eng. train | 29.10              | 32.90                 | 39.4                         | 39.90                           | 35.32         |

## Submission Platform
Participants will submit their predictions through the [CodaBench](https://www.codabench.org/competitions/18062/) platform, where their performance will be automatically evaluated and scored.

## Quickstart: Training & Submission

### 1. Train the Baseline Model
```bash
python main.py --config configs/baseline.yaml
```
- Trains the FOP dual-branch network with cross-entropy and Orthogonal Projection Loss (OPL).
- Automatically creates an open-set local validation split (10 unseen speakers) and evaluates true verification EER every epoch.
- Saves the best checkpoint to `output/checkpoints/gated_english_2.00_best.pth.tar`.

### 2. Generate CodaBench / Challenge Submission
```bash
python evaluate.py --config configs/baseline.yaml --ckpt output/checkpoints/gated_english_2.00_best.pth.tar
```
- Evaluates across all 4 competition tracks (`no_gender/English_heard`, `no_gender/Bangla_unheard`, `gender/English_heard`, `gender/Bangla_unheard`).
- Automatically bundles all four score files into `output/submission.zip`.
- Upload `output/submission.zip` directly to [CodaBench Competition #18062](https://www.codabench.org/competitions/18062/).

## Hierarchy

```
.
├── configs/
│   └── baseline.yaml                 # Centralized configuration (paths, hyperparameters, etc.)
├── src/
│   ├── data/                         # Data loading & open-set local validation splitting
│   ├── models/                       # FOP model architecture (Dual-branch & Gated Fusion)
│   ├── losses/                       # Orthogonal Projection Loss (OPL)
│   ├── evaluation/                   # Evaluation metrics (Challenge EER/AUC + Stratified 10-Fold)
│   └── config.py                     # YAML configuration loader
├── legacy/                           # Preserved original baseline reference scripts
├── main.py                           # Training entry point
└── evaluate.py                       # Inference & CodaBench submission packaging
```



## Setup

We used Anaconda to set up the environment for our experiments:

```
python==3.10.0
```

[CUDA](https://developer.nvidia.com/cuda-toolkit-archive) and
[cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) setup:

- CUDA Toolkit 12.4
- cuDNN v9.x for CUDA 12.x

To install PyTorch with GPU support:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Remaining dependencies:

```bash
pip install -r requirements.txt

```
