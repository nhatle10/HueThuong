
"""
"Fusion and Orthogonal Projection for Improved Face-Voice Association"
Muhammad Saad Saeed and Muhammad Haris Khan and Shah Nawaz and Muhammad Haroon Yousaf and Alessio Del Bue
ICASSP 2022
"""

from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import pandas as pd
# from scipy import random                                  # <<< CHANGE 1
import random                                               # <<< CHANGE 1
from sklearn import preprocessing
import matplotlib
matplotlib.use('Agg')                                       # <<< CHANGE 10 (headless)
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import online_evaluation
from tqdm import tqdm
from utils_config import load_config
from retrieval_model import FOP

EPS_1E3 = 1e-3
EPS_1E6 = 1e-6

def read_train_data(cfg):
    print(f"Reading Train Faces from: {cfg.data.train.faces}")
    img_train = pd.read_csv(cfg.data.train.faces, header=None)
    train_label = np.asarray(img_train.iloc[:, -1])
    img_train = np.asarray(img_train.iloc[:, :cfg.data.train.face_dim], dtype=np.float32)

    print(f"Reading Train Voices from: {cfg.data.train.voices}")
    voice_train = pd.read_csv(cfg.data.train.voices, header=None)
    voice_label = np.asarray(voice_train.iloc[:, -1], dtype=np.int64)
    voice_train = np.asarray(voice_train.iloc[:, :cfg.data.train.voice_dim], dtype=np.float32)

    # Sanity check: Ensure row alignment
    n_bad = int(np.sum(voice_label != train_label.astype(np.int64)))
    if n_bad:
        raise SystemExit('Label columns disagree on %d of %d rows -- the face '
                         'and voice CSVs are not row-aligned.'
                         % (n_bad, len(train_label)))
    print('Label columns agree on all %d rows' % len(train_label))

    # Encode labels to 0..N-1
    le = preprocessing.LabelEncoder()
    train_label = le.fit_transform(train_label)

    # Shuffle
    combined = list(zip(img_train, voice_train, train_label))
    random.shuffle(combined)
    img_train[:], voice_train[:], train_label[:] = zip(*combined)

    return np.asarray(img_train), np.asarray(voice_train), np.asarray(train_label)

def get_batch(batch_index, batch_size, labels, f_lst):
    start_ind = batch_index * batch_size
    end_ind = (batch_index + 1) * batch_size
    return np.asarray(f_lst[start_ind:end_ind]), np.asarray(labels[start_ind:end_ind])

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def main(face_train, voice_train, train_label, face_test, voice_test, cfg):
    device = torch.device('cuda' if cfg.model.cuda and torch.cuda.is_available() else 'cpu')
    print(f" + Training on device: {device}")
    
    n_class = int(np.max(train_label)) + 1
    print('  + n_class: %d' % n_class)
    
    model = FOP(cfg.model, face_train.shape[1], voice_train.shape[1], n_class)
    model.apply(init_weights)
    model.to(device)

    ce_loss = nn.CrossEntropyLoss().to(device)
    opl_loss = OrthogonalProjectionLoss(device)

    if device.type == 'cuda':
        # Faster training since PyTorch 1.12
        cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True
    
    # Optimizer parameters based on fusion type ('linear' and 'gated')
    if cfg.model.fusion == 'linear':
        parameters = [
                      {'params' : model.face_branch.fc1.parameters()},
                      {'params' : model.voice_branch.fc1.parameters()},
                      {'params': model.logits_layer.parameters()},
                        {'params' : model.fusion_layer.weight1},
                        {'params' : model.fusion_layer.weight2}]
    elif cfg.model.fusion == 'gated':
        parameters = [
                      {'params' : model.face_branch.fc1.parameters()},
                      {'params' : model.voice_branch.fc1.parameters()},
                      {'params': model.logits_layer.parameters()},
                      {'params' : model.fusion_layer.attention.parameters()}]
    
    optimizer = optim.Adam(parameters, lr=cfg.train.lr, weight_decay=getattr(cfg.train, 'weight_decay', EPS_1E3))
    
    os.makedirs('output', exist_ok=True)
    results = []
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    for alpha in cfg.train.alpha_list:
        eer_list = []
        epoch = 1
        num_of_batches = len(train_label) // cfg.train.batch_size
        loss_plot, auc_list = [], []
        loss_per_epoch = 0

        save_dir = os.path.join(cfg.run.save_dir, f"{cfg.model.fusion}_{cfg.run.tag}_alpha_{alpha:0.2f}")
        save_best = os.path.join(cfg.run.save_dir, f"best_{cfg.model.fusion}_{cfg.run.tag}_alpha_{alpha:0.2f}")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_best, exist_ok=True)

        txt = f"output/{cfg.model.fusion}_{cfg.run.tag}_ce_opl_{cfg.train.max_num_epoch:03d}_{alpha:0.2f}.txt"
        with open(txt, 'w+') as f:
            f.write('EPOCH\tLOSS\tVAL_EER\tVAL_AUC\tBEST_EER\n')

        best_eer = 1.0; best_epoch = 0; since_improved = 0
        with open(txt, 'a+') as f:
            while epoch <= cfg.train.max_num_epoch:
                print('Epoch %03d' % epoch)
                
                for idx in range(num_of_batches):
                    face_feats, batch_labels = get_batch(idx, cfg.train.batch_size, train_label, face_train)
                    voice_feats, _ = get_batch(idx, cfg.train.batch_size, train_label, voice_train)

                    loss_tmp, loss_opl, loss_soft, _, _ = train(
                        face_feats, voice_feats, batch_labels,
                        model, optimizer, ce_loss, opl_loss, alpha, device
                    )
                    loss_per_epoch += loss_tmp
                    
                loss_per_epoch = loss_per_epoch / num_of_batches
                loss_plot.append(loss_per_epoch)
                
                save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict()}, save_dir, f"checkpoint_{epoch:04d}.pth.tar")
                
                print('==> Epoch: %d/%d Loss: %0.2f Alpha:%0.2f' % (epoch, cfg.train.max_num_epoch, loss_per_epoch, alpha))
                
                eer, auc = online_evaluation.test(cfg, model, face_test, voice_test)
                eer_list.append(eer)
                auc_list.append(auc)

                if eer < best_eer - cfg.train.min_delta:
                    best_eer = eer; best_epoch = epoch; since_improved = 0
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'val_eer': eer, 'val_auc': auc,
                        'alpha': alpha, 'n_class': n_class,
                        'tag': cfg.run.tag
                    }, save_best, 'checkpoint_best.pth.tar')
                else:
                    since_improved += 1

                print('    val EER %.4f  val AUC %.4f  best %.4f @%d  patience %d/%d'
                      % (eer, auc, best_eer, best_epoch, since_improved, cfg.train.patience))
                epoch += 1
                
                f.write(f"{epoch - 1:04d}\t{loss_per_epoch:0.4f}\t{eer:0.4f}\t{auc:0.4f}\t{best_eer:0.4f}\n")
                f.flush()
                loss_per_epoch = 0
                
                if since_improved >= cfg.train.patience:
                    print(f"Early stop at epoch {epoch - 1}: Best val EER {best_eer:.4f} at epoch {best_epoch}.")
                    break
        
            # Save plots
        plt.figure(1); plt.title(f"Total Loss_{alpha:f}"); plt.plot(loss_plot)
        plt.savefig(f"output/{cfg.model.fusion}_{cfg.run.tag}_{alpha:0.2f}_total_loss.jpg", dpi=800); plt.clf()

        plt.figure(2); plt.title(f"EER_{alpha:f}"); plt.plot(eer_list)
        plt.savefig(f"output/{cfg.model.fusion}_{cfg.run.tag}_{alpha:0.2f}_eer.jpg", dpi=800); plt.clf()

        plt.figure(3); plt.title(f"AUC_{alpha:f}"); plt.plot(auc_list)
        plt.savefig(f"output/{cfg.model.fusion}_{cfg.run.tag}_{alpha:0.2f}_auc.jpg", dpi=800); plt.clf()

        results.append((alpha, best_eer, max(auc_list), best_epoch))

    print(f"\n{'='*60}\nSUMMARY ({cfg.run.tag})\n{'='*60}")
    print('%-8s%-12s%-12s%s' % ('alpha', 'val_EER', 'val_AUC', 'best_epoch'))
    
    for a, e, u, ep in results:
        print('%-8.2f%-12.4f%-12.4f%d' % (a, e, u, ep))

    best = min(results, key=lambda r: r[1])
    print(f"\nLowest val EER: alpha {best[0]:.2f}, {best[1]:.4f} at epoch {best[3]}")
    return loss_plot, best[1], best[2]    # <<< CHANGE 15
    
class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, device):
        super(OrthogonalProjectionLoss, self).__init__()
        self.device = device

    def forward(self, features, labels=None):
        features = F.normalize(features, p=2, dim=1)
        labels = labels[:, None]

        mask = torch.eq(labels, labels.t()).bool().to(self.device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(self.device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + EPS_1E6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + EPS_1E6)

        loss = (1.0 - pos_pairs_mean) + (0.7 * neg_pairs_mean)
        return loss, pos_pairs_mean, neg_pairs_mean


def train(face_feats, voice_feats, labels, model, optimizer, ce_loss, opl_loss, alpha, device):
    average_loss = RunningAverage()
    soft_losses = RunningAverage()
    opl_losses = RunningAverage()

    model.train()
    face_feats = torch.from_numpy(face_feats).float().to(device)
    voice_feats = torch.from_numpy(voice_feats).float().to(device)
    labels = torch.from_numpy(labels).to(device)

    face_feats, voice_feats, labels = Variable(face_feats), Variable(voice_feats), Variable(labels)
    comb, face_embeds, voice_embeds = model.train_forward(face_feats, voice_feats, labels)
    
    loss_opl, s_fac, d_fac = opl_loss(comb[0], labels)
    loss_soft = ce_loss(comb[1], labels)
    loss = loss_soft + alpha * loss_opl

    optimizer.zero_grad()
    loss.backward()
    
    average_loss.update(loss.item())
    opl_losses.update(loss_opl.item())
    soft_losses.update(loss_soft.item())
    optimizer.step()

    return average_loss.avg(), opl_losses.avg(), soft_losses.avg(), s_fac, d_fac

class RunningAverage(object):
    def __init__(self):
        self.value_sum = 0.
        self.num_items = 0. 

    def update(self, val):
        self.value_sum += val 
        self.num_items += 1

    def avg(self):
        average = 0.
        if self.num_items > 0:
            average = self.value_sum / self.num_items

        return average

def save_checkpoint(state, directory, filename):
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FLAG 2027 Baseline Training")
    parser.add_argument('--config', type=str, default='configs/baseline.yaml', help='Path to configuration YAML')
    cli_args, unparsed = parser.parse_known_args()

    cfg = load_config(cli_args.config)

    # Set seeds
    torch.manual_seed(cfg.run.seed)
    random.seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)
    if cfg.model.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.run.seed)

    print(f"[cfg] tag={cfg.run.tag} save_dir={cfg.run.save_dir} alphas={cfg.train.alpha_list} "
          f"batch={cfg.train.batch_size} patience={cfg.train.patience}")

    face_train, voice_train, train_label = read_train_data(cfg)
    face_test, voice_test = online_evaluation.read_data_from_config(cfg, track="no_gender", language="Bangla")

    loss_tmp, eer_tmp, auc_tmp = main(face_train, voice_train, train_label, face_test, voice_test, cfg)
