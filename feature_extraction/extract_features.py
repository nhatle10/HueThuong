#!/usr/bin/env python3
"""
extract_features.py -- extra face/voice backbone features for the FLAG 2027 release.

    .venv/bin/python feature_extraction/extract_features.py --encoder ecapa2
    (encoders: ecapa2, resnet293, wavlm_large, clip_l14, clip_h14, siglip2, agegender_vit)

Row contract (same as the existing VGGFace/ECAPA CSVs, so main.py/evaluate.py need no changes):
* train: row i = line i of train_set/train_English.txt, features + class id (field 6)
* dev:   row i = line i of <track>/<lang>_test.txt, features only

For each encoder two complete feature folders are written (the untouched modality is symlinked
from the original `features/` folder, so each folder can be used on its own):
    features_<enc>/       new encoder REPLACES the original modality
    features_<enc>_cat/   [original | new] CONCATENATED; the new block is centred (train mean), L2-normalised and scaled
                          to the mean row norm of the original block (train set) so neither dominates
Layout (relative to --data_root):
    train_set_extracted/train_set/features_<tag>/{faces,voices}/train_English_{faces,voices}.csv
    dev_set_extracted/dev_set/<track>/features_<tag>/<lang>_test_{faces,voices}.csv
A `.done` marker in train_set_extracted/train_set/features_<enc>/ marks a finished encoder.

Model weights (see scripts/extract_all.sh for downloads):
    ecapa2        feature_extraction/models/ecapa2.pt                        Jenthe/ECAPA2 (TorchScript)
    resnet293     feature_extraction/models/resnet293/voxceleb_resnet293_LM.onnx   WeSpeaker, VoxCeleb2
    wavlm_large   microsoft/wavlm-large (HF cache)       mean over time, averaged over all hidden layers
    clip_l14      openai/clip-vit-large-patch14 (HF cache)   projected image embedding
    clip_h14      laion/CLIP-ViT-H-14-laion2B-s32B-b79K (HF cache)   projected image embedding (1024)
    siglip2       google/siglip2-so400m-patch14-224 (HF cache)       pooled image embedding (1152)
    agegender_vit feature_extraction/models/agegender/pytorch_model.bin   abhilash88/age-gender-prediction,
                  ViT-B/16 backbone, CLS token (the "last shared layer" used by the FAME 2026 winner)
"""

import argparse
import os

import librosa
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

TRACKS = ("no_gender", "gender")
LANGS = ("English", "Bangla")
MODELS = "feature_extraction/models"


# ---------------------------------------------------------------- voice encoders (one clip at a time)
class VoiceEncoder:
    modality = "voices"
    max_seconds = None  # crop very long clips (memory); None = full clip

    def load_wav(self, path):
        wav, _ = librosa.load(path, sr=16000, mono=True)
        if self.max_seconds:
            wav = wav[: int(self.max_seconds * 16000)]
        return wav


class ECAPA2(VoiceEncoder):
    dim = 192

    def __init__(self, device):
        self.device = device
        self.model = torch.jit.load(f"{MODELS}/ecapa2.pt", map_location=device).eval()
        # TorchScript re-optimises during the first calls (outputs shift ~1e-3): warm up first
        with torch.inference_mode():
            for _ in range(3):
                self.model(torch.zeros(1, 16000 * 3, device=device))

    @torch.inference_mode()
    def embed(self, path):
        x = torch.from_numpy(self.load_wav(path)).unsqueeze(0).to(self.device)
        return self.model(x).squeeze(0).float().cpu().numpy()


class ResNet293(VoiceEncoder):
    dim = 256

    def __init__(self, device):
        import onnxruntime as ort
        import torchaudio.compliance.kaldi as kaldi

        self.kaldi = kaldi
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(16, os.cpu_count() or 4)
        self.sess = ort.InferenceSession(
            f"{MODELS}/resnet293/voxceleb_resnet293_LM.onnx", opts, providers=["CPUExecutionProvider"]
        )
        self.inp = self.sess.get_inputs()[0].name

    def embed(self, path):
        # WeSpeaker recipe: int16-scaled wav -> 80-dim Kaldi fbank -> per-utterance mean normalisation
        x = torch.from_numpy(self.load_wav(path)).unsqueeze(0) * (1 << 15)
        f = self.kaldi.fbank(x, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0,
                             sample_frequency=16000, window_type="hamming")
        f = f - f.mean(0, keepdim=True)
        return self.sess.run(None, {self.inp: f.unsqueeze(0).numpy()})[0][0]


class WavLMLarge(VoiceEncoder):
    dim = 1024
    max_seconds = 60

    def __init__(self, device):
        from transformers import AutoFeatureExtractor, WavLMModel

        self.device = device
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device).eval()

    @torch.inference_mode()
    def embed(self, path):
        inp = self.fe(self.load_wav(path), sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        hs = self.model(inp, output_hidden_states=True).hidden_states  # 25 x (1, T, 1024)
        return torch.stack([h.mean(1) for h in hs]).mean(0).squeeze(0).float().cpu().numpy()


# ---------------------------------------------------------------- face encoders (batched)
class FaceEncoder:
    modality = "faces"


class CLIPL14(FaceEncoder):
    dim = 768

    def __init__(self, device):
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        self.device = device
        self.proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

    @torch.inference_mode()
    def embed_batch(self, images):
        px = self.proc(images=images, return_tensors="pt").pixel_values.to(self.device)
        return self.model(pixel_values=px).image_embeds.float().cpu().numpy()


class CLIPH14(CLIPL14):
    """OpenCLIP ViT-H/14 (LAION-2B), transformers-format weights; projected image embedding."""
    dim = 1024

    def __init__(self, device):
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        repo = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        self.device = device
        self.proc = CLIPImageProcessor.from_pretrained(repo)
        self.model = CLIPVisionModelWithProjection.from_pretrained(repo).to(device).eval()


class SigLIP2(FaceEncoder):
    """SigLIP 2 So400m/14 at 224px (matches the 224px face crops); pooled image embedding."""
    dim = 1152

    def __init__(self, device):
        from transformers import AutoImageProcessor, AutoModel

        repo = "google/siglip2-so400m-patch14-224"
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(repo)
        self.model = AutoModel.from_pretrained(repo).to(device).eval()

    @torch.inference_mode()
    def embed_batch(self, images):
        px = self.proc(images=images, return_tensors="pt").pixel_values.to(self.device)
        out = self.model.get_image_features(pixel_values=px)
        out = getattr(out, "pooler_output", out)  # some transformers versions return a model output
        return out.float().cpu().numpy()


class AgeGenderViT(FaceEncoder):
    dim = 768

    def __init__(self, device):
        from transformers import ViTConfig, ViTImageProcessor, ViTModel

        self.device = device
        # The checkpoint uses pre-v5 ViT weight names; going through from_pretrained lets transformers
        # convert them. Keep only the ViT backbone (drop the age/gender heads) in a local HF folder once.
        hf_dir = f"{MODELS}/agegender/vit_backbone"
        if not os.path.exists(os.path.join(hf_dir, "pytorch_model.bin")):
            sd = torch.load(f"{MODELS}/agegender/pytorch_model.bin", map_location="cpu", weights_only=True)
            os.makedirs(hf_dir, exist_ok=True)
            ViTConfig().save_pretrained(hf_dir)  # ViT-B/16, 224px (the repo config only says model_type=vit)
            torch.save({k[len("vit."):]: v for k, v in sd.items() if k.startswith("vit.")},
                       os.path.join(hf_dir, "pytorch_model.bin"))
        self.model, info = ViTModel.from_pretrained(hf_dir, add_pooling_layer=False, output_loading_info=True)
        assert not info["missing_keys"], f"age-gender ViT: weights not loaded for {info['missing_keys']}"
        self.model.to(device).eval()
        self.proc = ViTImageProcessor(do_resize=True, size={"height": 224, "width": 224},
                                      image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])

    @torch.inference_mode()
    def embed_batch(self, images):
        px = self.proc(images=images, return_tensors="pt").pixel_values.to(self.device)
        return self.model(pixel_values=px).last_hidden_state[:, 0].float().cpu().numpy()


ENCODERS = {
    "ecapa2": ECAPA2,
    "resnet293": ResNet293,
    "wavlm_large": WavLMLarge,
    "clip_l14": CLIPL14,
    "clip_h14": CLIPH14,
    "siglip2": SigLIP2,
    "agegender_vit": AgeGenderViT,
}


# ---------------------------------------------------------------- extraction
def embed_all(enc, paths, desc, batch_size):
    out = np.zeros((len(paths), enc.dim), dtype=np.float32)
    n_fail = 0
    if enc.modality == "voices":
        for i, p in enumerate(tqdm(paths, desc=desc, mininterval=10)):
            try:
                out[i] = enc.embed(p)
            except Exception as e:  # keep the row contract: failed clips become zero vectors
                print(f"  FAILED {p}: {e}", flush=True)
                n_fail += 1
    else:
        for s in tqdm(range(0, len(paths), batch_size), desc=desc, mininterval=10):
            chunk = paths[s : s + batch_size]
            try:
                out[s : s + len(chunk)] = enc.embed_batch([Image.open(p).convert("RGB") for p in chunk])
            except Exception as e:
                print(f"  FAILED batch at {s}: {e}", flush=True)
                n_fail += len(chunk)
    if n_fail:
        print(f"  {n_fail} items failed and were written as zero vectors", flush=True)
    return out


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


def read_base(csv_path, has_label):
    vals = pd.read_csv(csv_path, header=None).values
    return (vals[:, :-1] if has_label else vals).astype(np.float32), (vals[:, -1].astype(int) if has_label else None)


def scaled(new, mean, target_norm):
    new = new - mean  # WavLM features share one large common direction until centred
    n = np.linalg.norm(new, axis=1, keepdims=True)
    return new / np.maximum(n, 1e-8) * target_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=sorted(ENCODERS))
    ap.add_argument("--data_root", default="data/FLAG_Grand_Challenge_2027_15_09_2027")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=64, help="face encoders only")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only first N rows per list, no outputs")
    args = ap.parse_args()

    enc = ENCODERS[args.encoder](args.device)
    mod = enc.modality
    other = "faces" if mod == "voices" else "voices"
    path_field = {"voices": (2, 1), "faces": (3, 2)}[mod]  # column in (train list, dev list)
    tag, tag_cat = args.encoder, f"{args.encoder}_cat"

    # ---- train
    tr_root = os.path.join(args.data_root, "train_set_extracted", "train_set")
    lst = pd.read_csv(os.path.join(tr_root, "train_set", "train_English.txt"), sep=r"\s+", header=None)
    if args.limit:
        lst = lst.head(args.limit)
    paths = [os.path.join(tr_root, "train_set", p) for p in lst[path_field[0]]]
    labels = lst[5].astype(int).values
    feats = embed_all(enc, paths, f"{tag} train", args.batch_size)
    if args.limit:
        print("smoke test OK:", feats.shape, "row norms", np.linalg.norm(feats, axis=1)[:5])
        return

    base_csv = os.path.join(tr_root, "features", mod, f"train_English_{mod}.csv")
    base, base_labels = read_base(base_csv, has_label=True)
    assert len(base) == len(feats) and (base_labels == labels).all(), "train list disagrees with existing CSV"
    target_norm = float(np.linalg.norm(base, axis=1).mean())
    new_mean = feats.mean(0, keepdims=True)  # train-set mean of the new block, reused for dev
    print(f"  original {mod} mean row norm on train: {target_norm:.3f}", flush=True)

    other_csv = os.path.join(tr_root, "features", other, f"train_English_{other}.csv")
    for t, data in ((tag, feats), (tag_cat, np.hstack([base, scaled(feats, new_mean, target_norm)]))):
        write_csv(os.path.join(tr_root, f"features_{t}", mod, f"train_English_{mod}.csv"), data, labels)
        symlink(other_csv, os.path.join(tr_root, f"features_{t}", other, f"train_English_{other}.csv"))

    # ---- dev
    dev_root = os.path.join(args.data_root, "dev_set_extracted", "dev_set")
    for track in TRACKS:
        for lang in LANGS:
            tdir = os.path.join(dev_root, track)
            lst = pd.read_csv(os.path.join(tdir, f"{lang}_test.txt"), sep=r"\s+", header=None)
            feats = embed_all(enc, [os.path.join(tdir, p) for p in lst[path_field[1]]], f"{tag} {track} {lang}",
                              args.batch_size)
            base, _ = read_base(os.path.join(tdir, "features", f"{lang}_test_{mod}.csv"), has_label=False)
            assert len(base) == len(feats), f"{track}/{lang}: list disagrees with existing CSV"
            for t, data in ((tag, feats), (tag_cat, np.hstack([base, scaled(feats, new_mean, target_norm)]))):
                write_csv(os.path.join(tdir, f"features_{t}", f"{lang}_test_{mod}.csv"), data)
                symlink(os.path.join(tdir, "features", f"{lang}_test_{other}.csv"),
                        os.path.join(tdir, f"features_{t}", f"{lang}_test_{other}.csv"))

    open(os.path.join(tr_root, f"features_{tag}", ".done"), "w").close()
    print(f"DONE {tag}: dims {enc.dim} (replace) / {base.shape[1] + enc.dim} (concat)", flush=True)


if __name__ == "__main__":
    main()
