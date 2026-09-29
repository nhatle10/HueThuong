
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

def main(face_train, voice_train, train_label, face_test, voice_test):
    n_class = int(np.max(train_label)) + 1
    print('  + n_class: %d' % n_class)
    model = FOP(FLAGS, face_train.shape[1], voice_train.shape[1], n_class)
    model.apply(init_weights)
    
    ce_loss = nn.CrossEntropyLoss()
    opl_loss = OrthogonalProjectionLoss()
    
    if FLAGS.cuda:
        model.cuda()
        ce_loss.cuda()    
        opl_loss.cuda()
        cudnn.benchmark = True
    
# =============================================================================
#     For Linear Fusion
# =============================================================================
    
    if FLAGS.fusion == 'linear':
    
        parameters = [
                      {'params' : model.face_branch.fc1.parameters()},
                      {'params' : model.voice_branch.fc1.parameters()},
                      {'params': model.logits_layer.parameters()},
                        {'params' : model.fusion_layer.weight1},
                        {'params' : model.fusion_layer.weight2}]
    
    
# =============================================================================
#     For Gated Fusion
# =============================================================================
    
    elif FLAGS.fusion == 'gated':
    
        parameters = [
                      {'params' : model.face_branch.fc1.parameters()},
                      {'params' : model.voice_branch.fc1.parameters()},
                      {'params': model.logits_layer.parameters()},
                      {'params' : model.fusion_layer.attention.parameters()}]

    optimizer = optim.Adam(parameters, lr=FLAGS.lr, weight_decay=0.01)

    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    os.makedirs('output', exist_ok=True)                              # <<< CHANGE 14
    results = []                                                      # <<< CHANGE 15

    for alpha in FLAGS.alpha_list:
        eer_list = []
        epoch=1
        num_of_batches = (len(train_label) // FLAGS.batch_size)
        loss_plot = []
        auc_list = []
        loss_per_epoch = 0
        # save_dir = '%s_%s_alpha_%0.2f'%(FLAGS.fusion, FLAGS.save_dir, alpha)
        save_dir = os.path.join(FLAGS.save_dir,                                   # <<< CHANGE 7
                                '%s_%s_alpha_%0.2f'%(FLAGS.fusion, FLAGS.tag, alpha))

        # txt = 'output/%s_ce_opl_%03d_%0.2f.txt'%(FLAGS.fusion, FLAGS.max_num_epoch, alpha)   # <<< CHANGE 14
        txt = 'output/%s_%s_ce_opl_%03d_%0.2f.txt'%(FLAGS.fusion, FLAGS.tag,      # <<< CHANGE 14
                                                    FLAGS.max_num_epoch, alpha)
        
        with open(txt,'w+') as f:
            # f.write('EPOCH\tLOSS\tEER\tAUC\n')                                  # <<< CHANGE 8
            f.write('EPOCH\tLOSS\tVAL_EER\tVAL_AUC\tBEST_EER\n')                  # <<< CHANGE 8
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # save_best = 'best_%s'%(save_dir)
        save_best = os.path.join(FLAGS.save_dir,                                  # <<< CHANGE 7
                                 'best_%s_%s_alpha_%0.2f'%(FLAGS.fusion, FLAGS.tag, alpha))
        
        # if not os.path.exists(save_best):                                       # <<< CHANGE 7
        #     os.mkdir(save_best)          # os.mkdir cannot create parent dirs
        os.makedirs(save_best, exist_ok=True)                                     # <<< CHANGE 7

        min_eer, max_auc = 1.0, 0.0                                               # <<< CHANGE 8
        best_eer = 1.0; best_epoch = 0; since_improved = 0                        # <<< CHANGE 8

        with open(txt,'a+') as f:
            # while (epoch < FLAGS.max_num_epoch):        # ran one epoch short   # <<< CHANGE 8
            while (epoch <= FLAGS.max_num_epoch):                                 # <<< CHANGE 8
                print('Epoch %03d'%(epoch))
                for idx in range(num_of_batches):
                    face_feats, batch_labels = get_batch(idx, FLAGS.batch_size, train_label, face_train)
                    voice_feats, _ = get_batch(idx, FLAGS.batch_size, train_label, voice_train)
                    loss_tmp, loss_opl, loss_soft, _, _ = train(face_feats, voice_feats, 
                                                                 batch_labels, 
                                                                 model, optimizer, ce_loss, opl_loss, alpha)
                    loss_per_epoch+=loss_tmp
                loss_per_epoch = loss_per_epoch/num_of_batches
                loss_plot.append(loss_per_epoch)
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict()}, save_dir, 'checkpoint_%04d.pth.tar'%(epoch))
                print('==> Epoch: %d/%d Loss: %0.2f Alpha:%0.2f'%(epoch, FLAGS.max_num_epoch, loss_per_epoch, alpha))
                
                eer, auc = online_evaluation.test(FLAGS, model, face_test, voice_test)
                eer_list.append(eer)
                auc_list.append(auc)

                # <<< CHANGE 8: was  if eer <= min(eer_list):  -- eer_list already
                # contains the current value, so `<=` re-saved on every plateau.
                # A strict improvement of at least --min_delta now counts.
                # if eer <= min(eer_list):
                #     min_eer = eer
                #     max_auc = auc
                #     save_checkpoint({
                #     'epoch': epoch,
                #     'state_dict': model.state_dict()}, save_best, 'checkpoint_%04d.pth.tar'%(epoch))
                if eer < best_eer - FLAGS.min_delta:                              # <<< CHANGE 8
                    best_eer = eer; best_epoch = epoch; since_improved = 0
                    min_eer = eer
                    max_auc = auc
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'val_eer': eer, 'val_auc': auc,
                        'alpha': alpha, 'n_class': n_class,
                        'tag': FLAGS.tag},
                        save_best, 'checkpoint_best.pth.tar')
                else:                                                             # <<< CHANGE 8
                    since_improved += 1

                print('    val EER %.4f  val AUC %.4f  best %.4f @%d  patience %d/%d'
                      % (eer, auc, best_eer, best_epoch, since_improved, FLAGS.patience))

                epoch += 1
                # f.write('%04d\t%0.4f\t%0.2f\t%0.2f\n'%(epoch, loss_per_epoch, eer, auc))   # <<< CHANGE 8
                f.write('%04d\t%0.4f\t%0.4f\t%0.4f\t%0.4f\n'                      # <<< CHANGE 8
                        % (epoch - 1, loss_per_epoch, eer, auc, best_eer))
                f.flush()                                                         # <<< CHANGE 8
                loss_per_epoch = 0

                if since_improved >= FLAGS.patience:                              # <<< CHANGE 8
                    print('Early stop at epoch %d: no val improvement > %.4g for '
                          '%d epochs. Best val EER %.4f at epoch %d.'
                          % (epoch - 1, FLAGS.min_delta, FLAGS.patience,
                             best_eer, best_epoch))
                    break
        
        plt.figure(1)
        plt.title('Total Loss_%f'%(alpha))
        plt.plot(loss_plot)
        plt.savefig('output/%s_%s_%0.2f_total_loss.jpg'%(FLAGS.fusion, FLAGS.tag, alpha), dpi=800)
        plt.clf()                                                                 # <<< CHANGE 15
        
        plt.figure(2)
        plt.title('EER_%f'%(alpha))
        plt.plot(eer_list)
        plt.savefig('output/%s_%s_%0.2f_eer.jpg'%(FLAGS.fusion, FLAGS.tag, alpha), dpi=800)
        plt.clf()                                                                 # <<< CHANGE 15
        
        plt.figure(3)
        plt.title('AUC_%f'%(alpha))
        plt.plot(auc_list)
        plt.savefig('output/%s_%s_%0.2f_auc.jpg'%(FLAGS.fusion, FLAGS.tag, alpha), dpi=800)
        plt.clf()                                                                 # <<< CHANGE 15

        print('  best checkpoint: %s/checkpoint_best.pth.tar (epoch %d, val EER %.4f)'
              % (save_best, best_epoch, best_eer))
        results.append((alpha, best_eer, max_auc, best_epoch))                    # <<< CHANGE 15

        # <<< CHANGE 15: this `return` sat INSIDE the alpha loop, so only the
        # first alpha ever ran. Dedented to after the loop.
        # return loss_plot, min_eer, max_auc

    print('\n%s\nSUMMARY (%s)\n%s' % ('=' * 60, FLAGS.tag, '=' * 60))             # <<< CHANGE 15
    print('%-8s%-12s%-12s%s' % ('alpha', 'val_EER', 'val_AUC', 'best_epoch'))
    for a, e, u, ep in results:
        print('%-8.2f%-12.4f%-12.4f%d' % (a, e, u, ep))
    best = min(results, key=lambda r: r[1])
    print('\nLowest val EER: alpha %.2f, %.4f at epoch %d' % (best[0], best[1], best[3]))
    print('Select this alpha and checkpoint, THEN evaluate on test.')
    return loss_plot, best[1], best[2]                                            # <<< CHANGE 15
    
class OrthogonalProjectionLoss(nn.Module):
    def __init__(self):
        super(OrthogonalProjectionLoss, self).__init__()
        self.device = (torch.device('cuda') if FLAGS.cuda else torch.device('cpu'))

    def forward(self, features, labels=None):
        
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]

        mask = torch.eq(labels, labels.t()).bool().to(self.device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(self.device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

        loss = (1.0 - pos_pairs_mean) + (0.7 * neg_pairs_mean)

        return loss, pos_pairs_mean, neg_pairs_mean


def train(face_feats, voice_feats, labels, model, optimizer, ce_loss, opl_loss, alpha):
    
    average_loss = RunningAverage()
    soft_losses = RunningAverage()
    opl_losses = RunningAverage()

    model.train()
    face_feats = torch.from_numpy(face_feats).float()
    voice_feats = torch.from_numpy(voice_feats).float()
    labels = torch.from_numpy(labels)
    
    if FLAGS.cuda:
        face_feats, voice_feats, labels = face_feats.cuda(), voice_feats.cuda(), labels.cuda()

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

    class ConfigFlags:
        pass

    global FLAGS
    FLAGS = ConfigFlags()
    FLAGS.cuda = cfg.model.cuda and torch.cuda.is_available()
    FLAGS.fusion = cfg.model.fusion
    FLAGS.dim_embed = cfg.model.dim_embed
    FLAGS.lr = cfg.train.lr
    FLAGS.weight_decay = getattr(cfg.train, 'weight_decay', 0.01)
    FLAGS.batch_size = cfg.train.batch_size
    FLAGS.max_num_epoch = cfg.train.max_num_epoch
    FLAGS.alpha_list = cfg.train.alpha_list if isinstance(cfg.train.alpha_list, list) else [float(cfg.train.alpha_list)]
    FLAGS.patience = cfg.train.patience
    FLAGS.min_delta = cfg.train.min_delta
    FLAGS.save_dir = cfg.run.save_dir
    FLAGS.tag = cfg.run.tag
    FLAGS.seed = cfg.run.seed

    torch.manual_seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    if FLAGS.cuda:
        torch.cuda.manual_seed(FLAGS.seed)

    print('[cfg] tag=%s save_dir=%s alphas=%s batch=%d patience=%d'
          % (FLAGS.tag, FLAGS.save_dir, FLAGS.alpha_list, FLAGS.batch_size, FLAGS.patience))

    face_train, voice_train, train_label = read_train_data(cfg)
    face_test, voice_test = online_evaluation.read_data_from_config(cfg, track="no_gender", language="Bangla")

    loss_tmp, eer_tmp, auc_tmp = main(face_train, voice_train, train_label, face_test, voice_test)