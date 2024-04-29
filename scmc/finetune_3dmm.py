import argparse
import os
import json
import math
import time
import torch
import warnings
import librosa
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

from models import ExprEncoder, AudioEncoder, Loss, ScaleLayer, MDist
from cross_attention import CrossAttentionSeq



def load_audio(audio_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio, _ = librosa.load(audio_path, sr=16000)
        return audio


class AlignedDataset(Dataset):
    _emotions = ['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
    _levels = [1, 3, 3, 3, 3, 3, 3]
    _emo_2_lvl = {e:l for e, l in zip(_emotions, _levels)}
    
    def __init__(self, args):
        super().__init__()
        self.data_root = args.data_root
        with open(args.aligned_path) as f:
            self.aligned_path = json.load(f)
        self.actors = args.actors
        self.emotions = args.emotions
        self.bs = args.bs
        self.nframe = args.nframe
        self.fps = args.fps
        self.expr_clips = []
        self.audio_clips = []
        self.labels = []
        self.IDs = []

        cache_path = '.cache/finetune_expr_tmpr_dataset.pkl'
        if os.path.exists(cache_path):
            LOG('load data from cache')
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            self.expr_clips = cache_data['expr_clips']
            self.audio_clips = cache_data['audio_clips']
            self.labels = cache_data['labels']
            self.IDs = cache_data['IDs']
            LOG(f'{len(self.expr_clips)} samples loaded.')
        else:
            self.load_data()
            cache_data = {
                'expr_clips': self.expr_clips,
                'audio_clips': self.audio_clips,
                'labels': self.labels,
                'IDs': self.IDs
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

    def load_data(self):
        LOG('loading data...')
        for actor in self.actors:
            actor_root = os.path.join(
                self.data_root, '{}_deca_avcl.pkl'.format(actor))
            assert os.path.isfile(
                actor_root), '%s is not a valid file' % actor_root

            actor_data = pickle.load(open(actor_root, "rb"))

            for label, emo in enumerate(self.emotions):
                if emo == 'neutral':
                    continue
                for k, v in self.aligned_path[actor][emo].items():
                    src, dst = k.split('_')
                    path0, path1 = v
                    if src not in actor_data['neutral'] or dst not in actor_data[emo]:
                        continue
                    src_expr = actor_data['neutral'][src]
                    dst_expr = actor_data[emo][dst]
                    src_expr = np.concatenate((src_expr[:, 0:1], src_expr[:, 3:]), 1) # 51
                    dst_expr = np.concatenate((dst_expr[:, 0:1], dst_expr[:, 3:]), 1) # 51
                    src_audio = os.path.join(self.data_root, actor, 'audio/neutral/level_1', src+'.m4a')
                    dst_audio = os.path.join(self.data_root, actor, 'audio', emo, f'level_{self._emo_2_lvl[emo]}', dst+'.m4a')
                    src_audio = load_audio(src_audio)
                    dst_audio = load_audio(dst_audio)
                    frame_len = 16000 // self.fps
                    for i in range(len(path0)-self.nframe+1):
                        fid0 = path0[i]
                        fid1 = path1[i]
                        audio_src_batch = []
                        audio_dst_batch = []
                        for j in range(self.nframe):
                            start = min(len(src_audio)-frame_len,
                                        (fid0+j)*frame_len)
                            end = min(len(src_audio), (fid0+j+1)*frame_len)
                            src_clip = src_audio[start:end]
                            start = min(len(dst_audio)-frame_len,
                                        (fid1+j)*frame_len)
                            end = min(len(dst_audio), (fid1+j+1)*frame_len)
                            dst_clip = dst_audio[start:end]
                            audio_src_batch.append(src_clip)
                            audio_dst_batch.append(dst_clip)

                        expr_src_batch = []
                        expr_dst_batch = []
                        for j in range(self.nframe):
                            if fid0+j >= src_expr.shape[0] or fid1+j >= dst_expr.shape[0]:
                                break
                            expr_src_batch.append(src_expr[fid0+j])
                            expr_dst_batch.append(dst_expr[fid1+j])
                        
                        expr_batch = expr_src_batch + expr_dst_batch
                        if len(expr_batch) == self.nframe * 2:
                            self.audio_clips.append(np.stack(audio_src_batch+audio_dst_batch, axis=0)[:, np.newaxis, :])
                            self.expr_clips.append(np.stack(expr_batch, axis=0)[:, np.newaxis, :])
                            self.labels.append(label)
                            self.IDs.append(actor)
        LOG(f'{len(self.expr_clips)} samples loaded.')

    def __len__(self):
        return len(self.expr_clips)
    
    def __getitem__(self, index):
        expr_batch = [self.expr_clips[index]]
        audio_batch = [self.audio_clips[index]]

        while len(expr_batch) < self.bs:
            j = np.random.randint(0, len(self.expr_clips))
            if j == index or self.labels[j] != self.labels[index] or self.IDs[j] != self.IDs[index]:
                continue
            expr_batch.append(self.expr_clips[j])
            audio_batch.append(self.audio_clips[j])

        return np.stack(expr_batch, axis=0).astype(np.float32),  np.stack(audio_batch, axis=0).astype(np.float32)


def LOG(*args, **kwargs):
    _LOG(time.strftime('%Y-%m-%d#%H:%M:%S',time.localtime(time.time())), end=' ')
    _LOG(*args, **kwargs)
    log_file.flush()


def _LOG(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_file)


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    expr_encoder = ExprEncoder(args.depth1, args.dim).to(device).train()
    expr_ckpt = torch.load(args.expr_ckpt, device)['model']
    expr_encoder.load_state_dict(expr_ckpt)
    audio_encoder = AudioEncoder(args.depth2, args.dim).to(device).train()
    audio_ckpt = torch.load(args.audio_ckpt, device)['model']
    audio_encoder.load_state_dict(audio_ckpt)
    expr_attn = CrossAttentionSeq(
        num_blocks=args.num_blocks, 
        in_features=expr_encoder.feat_dim, 
        dim=args.dim, heads=4).to(device).train()
    audio_attn = CrossAttentionSeq(
        num_blocks=args.num_blocks, 
        in_features=audio_encoder.feat_dim, 
        dim=args.dim, heads=4).to(device).train()
    scale_layer = ScaleLayer().to(device).train()
    loss_fn = Loss(args.loss_mode, device)
    audio_mdist = MDist(args.loss_mode)
    expr_mdist = MDist(args.loss_mode)
    dataset = AlignedDataset(args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, persistent_workers=True)
    optimizer = torch.optim.AdamW(
        list(audio_encoder.parameters()) +
        list(expr_encoder.parameters()) +
        list(scale_layer.parameters()) +
        list(expr_attn.parameters()) +
        list(audio_attn.parameters()),
        lr=args.lr, weight_decay=args.wd)
    
    for epoch in range(args.epochs):
        for step, (exprs, audios) in enumerate(dataloader):
            _, bs, nframe, c, l = exprs.shape
            exprs = exprs.view(bs*nframe, c, l).to(device)
            _, bs, nframe, c, l = audios.shape
            audios = audios.view(bs*nframe, c, l).to(device)

            expr_vec = expr_encoder.encode(exprs)
            expr_vec = expr_vec.view(bs, nframe, *expr_vec.shape[1:])
            fused_expr_vec = [expr_vec[:, 0], expr_vec[:, nframe//2]]
            for i in range(1, args.nframe):
                fused_expr_vec[0] = expr_attn(
                    expr_vec[:, i], fused_expr_vec[0])
                fused_expr_vec[1] = expr_attn(
                    expr_vec[:, nframe//2+i], fused_expr_vec[1])
            fused_expr_vec[0] = expr_encoder.activate(fused_expr_vec[0])
            fused_expr_vec[1] = expr_encoder.activate(fused_expr_vec[1])
            fused_expr_vec = torch.stack(fused_expr_vec, dim=1)
            loss_expr, expr_dist = loss_fn(fused_expr_vec)
            if step > 5000:
                expr_dist = scale_layer(expr_dist)

            audio_vec = audio_encoder.encode(audios)
            audio_vec = audio_vec.view(bs, nframe, *audio_vec.shape[1:])
            fused_audio_vec = [audio_vec[:, 0], audio_vec[:, nframe//2]]
            for i in range(1, args.nframe):
                fused_audio_vec[0] = audio_attn(
                    audio_vec[:, i], fused_audio_vec[0])
                fused_audio_vec[1] = audio_attn(
                    audio_vec[:, nframe//2+i], fused_audio_vec[1])
            fused_audio_vec[0] = audio_encoder.activate(fused_audio_vec[0])
            fused_audio_vec[1] = audio_encoder.activate(fused_audio_vec[1])
            fused_audio_vec = torch.stack(fused_audio_vec, dim=1)
            loss_audio, audio_dist = loss_fn(fused_audio_vec)
            
            loss_consis = args.lambd_consis * \
                torch.abs(expr_dist - audio_dist).mean() if step > 5000 else audio_dist.new_zeros(1)
            loss = loss_audio + loss_expr + loss_consis
            loss.backward()
            optimizer.step()

            expr_mdist.update(fused_expr_vec, scale_layer)
            audio_mdist.update(fused_audio_vec)

            if (step + 1) % args.log_interval == 0:
                LOG(f'epoch {epoch+1}, step [{step+1}/{len(dataloader)}], loss_audio {loss_audio.item():.4f}, loss_expr {loss_expr.item():.4f}, ' 
                    f'loss_consis {loss_consis.item():.4f}, expr_dist ({expr_mdist.pos_dist:.4f}, {expr_mdist.neg_dist:.4f}), '
                    f'audio_dist ({audio_mdist.pos_dist:.4f}, {audio_mdist.neg_dist:.4f})')
        
        ckpt = {
            'model': {
                'audio_encoder': audio_encoder.state_dict(),
                'expr_encoder': expr_encoder.state_dict(),
                'scale_layer': scale_layer.state_dict(),
                'expr_attn': expr_attn.state_dict(),
                'audio_attn': audio_attn.state_dict()
            },
            'optimizer': optimizer.state_dict()
        }
        torch.save(ckpt, os.path.join(args.work_dir, 'last.pth'))
    LOG('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth1',type=int,default=14, help='expr encoder depth')
    parser.add_argument('--depth2',type=int,default=14, help='audio encoder depth')
    parser.add_argument('--dim',type=int,default=128)
    parser.add_argument('--num_blocks', type=int, default=1, help='num blocks for cross attention')
    parser.add_argument('--expr_ckpt',type=str,default='exp/similarity/3dmm/last.pth', help='expr encoder checkpoint path')
    parser.add_argument('--audio_ckpt',type=str,default='exp/similarity/audio/last.pth', help='audio encoder checkpoint path')
    parser.add_argument('--data_root',type=str,default='MEAD', help='data root')
    parser.add_argument('--aligned_path',type=str,default='MEAD/aligned_path.json', help='aligned_path.json')
    parser.add_argument('--actors', type=str, nargs='+', help='Subset of the MEAD actors', default=['M035', 'M023', 'W026', 'W025', 'W021', 'W019', 'M040', 'M011', 'W038', 'W023', 'W033', 'W040', 'M032', 'W036', 'M022', 'M039', 'W035', 'W016', 'M041', 'M027', 'M031', 'W014', 'M005', 'M019', 'M025', 'M042', 'M028', 'M037', 'M033', 'M024', 'W011', 'W028', 'W018', 'M034', 'M029', 'M007'])
    parser.add_argument('--emotions', type=str, nargs='+', help='Selection of Emotions', default=['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised'])
    parser.add_argument('--fps',type=int,default=30, help='video fps')
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--bs',type=int,default=64)
    parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--lr',type=float,default=0.00001)
    parser.add_argument('--wd',type=float,default=0., help='weight decay')
    parser.add_argument('--loss_mode', type=str, default='l2')
    parser.add_argument('--nframe',type=int,default=1, help='num frames for positive samples')
    parser.add_argument('--lambd_consis',type=float,default=1, help='consistency loss weight')
    parser.add_argument('--work_dir',type=str,default='exp/similarity/3dmm_finetune')
    parser.add_argument('--log_interval',type=int,default=50)
    parser.add_argument('--seed',type=int,default=2024)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.work_dir, exist_ok=True)
    log_file = open(os.path.join(args.work_dir, 'loss_log.txt'), 'w')
    
    main()

    log_file.close()
