# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

import os
import argparse
# import fairseq # 去死吧fairseq
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from s3prl.nn import S3PRLUpstream
import random

from icecream import ic # 太乱了换个显示区分一下
import schedulefree # 还得是你
random.seed(1984)

ssl_model_list = [
    "wavlm_base",
    "wavlm_large",
    "wav2vec2_base_960",
    "wav2vec2_large_lv60_cv_swbd_fsh",
    "hubert_base",
    "hubert_large_ll60k",
    "xls_r_300m",
]

def load_ssl_model_s3prl(ssl_model_type, use_proxy = True):
    assert ssl_model_type in ssl_model_list, (
        f"***ERROR***: {ssl_model_type} is not support, please check ssl_model_list."
    )
    if "base" in ssl_model_type:
        SSL_OUT_DIM = 768
    elif "large" in ssl_model_type or ssl_model_type in ["xls_r_300m"]:
        SSL_OUT_DIM = 1024
    if use_proxy:
        os.environ['http_proxy'] = 'http://127.0.0.1:7890'
        os.environ['https_proxy'] = 'http://127.0.0.1:7890'
    ssl_model = S3PRLUpstream(ssl_model_type)
    return SSL_Model(ssl_model, SSL_OUT_DIM), SSL_OUT_DIM

class SSL_Model(nn.Module):
    def __init__(
        self, 
        ssl_model, 
        ssl_out_dim,
    ):
        super(SSL_Model, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_out_dim = ssl_out_dim
        
    def forward(self, wav, wav_length):
        wav = wav.squeeze(1)  # [B, T]
        ssl_features, ssl_lens = self.ssl_model(wav, wav_length)
        return ssl_features[-1]

class MOS_Predictor(nn.Module):
    def __init__(self, ssl_model_type):
        super(MOS_Predictor, self).__init__()
        self.ssl_model, feature_dim = load_ssl_model_s3prl(ssl_model_type)
        self.linear = nn.Linear(feature_dim, 1)
        
    def forward(self, audio):
        # print(audio.shape)
        audio_length = torch.tensor([audio.shape[2]])
        x = self.ssl_model(audio, audio_length)
        frame_score = self.linear(x).squeeze(-1)
        utt_score = torch.mean(frame_score, dim=1)
        return utt_score

    
class MyDataset(Dataset):
    def __init__(self, wavdir, mos_list):
        self.mos_lookup = { }
        f = open(mos_list, 'r')
        for line in f:
            parts = line.strip().split(',')
            wavname = parts[0]
            mos = float(parts[1])
            self.mos_lookup[wavname] = mos

        self.wavdir = wavdir
        self.wavnames = sorted(self.mos_lookup.keys())

        
    def __getitem__(self, idx):
        wavname = self.wavnames[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
        score = self.mos_lookup[wavname]
        return wav, score, wavname
    

    def __len__(self):
        return len(self.wavnames)


    def collate_fn(self, batch):  ## zero padding
        wavs, scores, wavnames = zip(*batch)
        wavs = list(wavs)
        max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)

        output_wavs = torch.stack(output_wavs, dim=0)
        scores  = torch.stack([torch.tensor(x) for x in list(scores)], dim=0)
        return output_wavs, scores, wavnames

    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='Path of your DATA/ directory')
    # parser.add_argument('--fairseq_base_model', type=str, required=True, help='Path to pretrained fairseq base model')
    parser.add_argument('--finetune_from_checkpoint', type=str, required=False, help='Path to your checkpoint to finetune from')
    parser.add_argument('--outdir', type=str, required=False, default='checkpoints', help='Output directory for your trained checkpoints')
    args = parser.parse_args()

    # cp_path = args.fairseq_base_model
    datadir = args.datadir
    ckptdir = args.outdir
    my_checkpoint = args.finetune_from_checkpoint
    
    if not os.path.exists(ckptdir):
        os.system('mkdir -p ' + ckptdir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ic('DEVICE: ' + str(device))

    wavdir = os.path.join(datadir, 'wav')
    trainlist = os.path.join(datadir, 'sets/train_mos_list.txt')
    validlist = os.path.join(datadir, 'sets/val_mos_list.txt')

    # ssl_model_type = cp_path.split('/')[-1]
    # if ssl_model_type == 'wav2vec_small.pt':
    #     SSL_OUT_DIM = 768
    # elif ssl_model_type in ['w2v_large_lv_fsh_swbd_cv.pt', 'xlsr_53_56k.pt']:
    #     SSL_OUT_DIM = 1024
    # else:
    #     print('*** ERROR *** SSL model type ' + ssl_model_type + ' not supported.')
    #     exit()
    #
    # model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    # ssl_model = model[0]
    # ssl_model.remove_pretraining_modules()
    
    trainset = MyDataset(wavdir, trainlist)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2, collate_fn=trainset.collate_fn)

    validset = MyDataset(wavdir, validlist)
    validloader = DataLoader(validset, batch_size=2, shuffle=True, num_workers=2, collate_fn=validset.collate_fn)

    ssl_model_type = "wav2vec2_base_960"
    net = MOS_Predictor(ssl_model_type)
    net = net.to(device)

    if my_checkpoint != None:  ## do (further) finetuning
        net.load_state_dict(torch.load(my_checkpoint))
    
    criterion = nn.L1Loss()
    # optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    ic(len(trainloader))
    optimizer = schedulefree.SGDScheduleFree(net.parameters(), lr=0.001, momentum=0.9, warmup_steps=len(trainloader))

    PREV_VAL_LOSS=9999999999
    orig_patience=20
    patience=orig_patience
    for epoch in range(1,1001):
        STEPS=0
        net.train()
        optimizer.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels, filenames = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            STEPS += 1
            running_loss += loss.item()
            ic('STEPS:' + str(i))
            ic('TRAIN LOSS:' + str(loss))
        ic('EPOCH: ' + str(epoch))
        ic('AVG EPOCH TRAIN LOSS: ' + str(running_loss / STEPS))
        epoch_val_loss = 0.0
        net.eval()
        optimizer.eval()
        ## clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        ## validation
        VALSTEPS=0
        for i, data in enumerate(validloader, 0):
            VALSTEPS+=1
            inputs, labels, filenames = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            epoch_val_loss += loss.item()

        avg_val_loss=epoch_val_loss/VALSTEPS    
        ic('EPOCH VAL LOSS: ' + str(avg_val_loss))
        if avg_val_loss < PREV_VAL_LOSS:
            ic('Loss has decreased')
            PREV_VAL_LOSS=avg_val_loss
            PATH = os.path.join(ckptdir, 'ckpt_' + str(epoch))
            torch.save(net.state_dict(), PATH)
            patience = orig_patience
        else:
            patience-=1
            if patience == 0:
                ic('loss has not decreased for ' + str(orig_patience) + ' epochs; early stopping at epoch ' + str(epoch))
                break
        
    ic('Finished Training')

if __name__ == '__main__':
    main()
