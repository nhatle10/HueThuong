import argparse
import os
import numpy as np
import torch
from torch.autograd import Variable
import pandas as pd
from retrieval_model import FOP
from utils_config import load_config

def read_data(test_file_face, test_file_voice, face_dim=4096, voice_dim=192):
    print(f'Reading Test Face: {test_file_face}')
    face_test = pd.read_csv(test_file_face, header=None).values[:, :face_dim]
    print(f'Reading Test Voice: {test_file_voice}')
    voice_test = pd.read_csv(test_file_voice, header=None).values[:, :voice_dim]

    face_test = torch.from_numpy(face_test).float()
    voice_test = torch.from_numpy(voice_test).float()
    print('  face %s   voice %s' % (face_test.shape, voice_test.shape))
    return face_test, voice_test

def compute_and_save_scores(cfg, ckpt_path, track='no_gender', heard_lang='English'):
    unheard_lang = 'Bangla' if heard_lang == 'English' else 'English'
    dev_root = cfg.data.dev.root
    face_dim = cfg.data.train.face_dim
    voice_dim = cfg.data.train.voice_dim
    use_cuda = cfg.model.cuda and torch.cuda.is_available()

    print(f"\n{'='*50}\nEvaluating Track: {track} | Heard: {heard_lang} | Unheard: {unheard_lang}")
    
    # 1. Load data
    face_file_heard = os.path.join(dev_root, track, "features", f"{heard_lang}_test_faces.csv")
    voice_file_heard = os.path.join(dev_root, track, "features", f"{heard_lang}_test_voices.csv")
    face_test_heard, voice_test_heard = read_data(face_file_heard, voice_file_heard, face_dim, voice_dim)

    face_file_unheard = os.path.join(dev_root, track, "features", f"{unheard_lang}_test_faces.csv")
    voice_file_unheard = os.path.join(dev_root, track, "features", f"{unheard_lang}_test_voices.csv")
    face_test_unheard, voice_test_unheard = read_data(face_file_unheard, voice_file_unheard, face_dim, voice_dim)

    # 2. Build model and load checkpoint
    n_class = 70  # 70 training speakers in FLAG 2027
    model = FOP(cfg.model, face_dim, voice_dim, n_class)
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location='cuda' if use_cuda else 'cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    print(f"=> Loaded checkpoint '{ckpt_path}' (epoch {checkpoint.get('epoch', 'N/A')})")
    
    model.eval()
    if use_cuda:
        model.cuda()
        face_test_heard, voice_test_heard = face_test_heard.cuda(), voice_test_heard.cuda()
        face_test_unheard, voice_test_unheard = face_test_unheard.cuda(), voice_test_unheard.cuda()

    with torch.no_grad():
        _, face_h, voice_h = model(face_test_heard, voice_test_heard)
        _, face_u, voice_u = model(face_test_unheard, voice_test_unheard)

        face_h = face_h.cpu().detach().numpy()
        voice_h = voice_h.cpu().detach().numpy()
        face_u = face_u.cpu().detach().numpy()
        voice_u = voice_u.cpu().detach().numpy()

        scores_heard = np.linalg.norm(face_h - voice_h, axis=1)
        scores_unheard = np.linalg.norm(face_u - voice_u, axis=1)

    # 3. Read test keys
    key_heard_path = os.path.join(dev_root, track, f"{heard_lang}_test.txt")
    key_unheard_path = os.path.join(dev_root, track, f"{unheard_lang}_test.txt")

    with open(key_heard_path, 'r') as f:
        keys_heard = [line.strip().split()[0] for line in f if line.strip()]
    with open(key_unheard_path, 'r') as f:
        keys_unheard = [line.strip().split()[0] for line in f if line.strip()]

    assert len(keys_heard) == len(scores_heard), f"Heard mismatch: {len(keys_heard)} keys vs {len(scores_heard)} scores"
    assert len(keys_unheard) == len(scores_unheard), f"Unheard mismatch: {len(keys_unheard)} keys vs {len(scores_unheard)} scores"

    out_dir = os.path.join("output", "sub_score_v4", track)
    os.makedirs(out_dir, exist_ok=True)

    out_heard = os.path.join(out_dir, f"sub_score_v4_{heard_lang}_heard.txt")
    out_unheard = os.path.join(out_dir, f"sub_score_v4_{unheard_lang}_unheard.txt")

    with open(out_heard, 'w') as f:
        for k, s in zip(keys_heard, scores_heard):
            f.write(f"{k} {s:.6f}\n")

    with open(out_unheard, 'w') as f:
        for k, s in zip(keys_unheard, scores_unheard):
            f.write(f"{k} {s:.6f}\n")

    print(f"  Wrote: {out_heard} ({len(keys_heard)} pairs)")
    print(f"  Wrote: {out_unheard} ({len(keys_unheard)} pairs)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute Submission Scores")
    parser.add_argument('--config', type=str, default='configs/baseline.yaml', help='Path to config YAML')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint (.pth.tar)')
    parser.add_argument('--track', type=str, default='all', choices=['all', 'no_gender', 'gender'], help='Track to evaluate')
    parser.add_argument('--heard_lang', type=str, default='English', choices=['English', 'Bangla'], help='Heard language (trained on)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    tracks = ['no_gender', 'gender'] if args.track == 'all' else [args.track]

    for t in tracks:
        compute_and_save_scores(cfg, args.ckpt, track=t, heard_lang=args.heard_lang)