import argparse
import os
import librosa
import json
import math
import time
import torch
import warnings
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import AudioEncoder, Loss, MDist


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
        self.nframe = args.nframe
        self.bs = args.bs
        self.fps = args.fps
        self.audio_clips = []
        self.labels = []
        self.IDs = []

        LOG('loading data...')
        for actor in self.actors:
            for label, emo in enumerate(self.emotions):
                if emo == 'neutral':
                    continue

                for k, v in self.aligned_path[actor][emo].items():
                    src, dst = k.split('_')
                    path0, path1 = v
                    src_audio = os.path.join(self.data_root, actor, 'audio/neutral/level_1', src+'.m4a')
                    dst_audio = os.path.join(self.data_root, actor, 'audio', emo, f'level_{self._emo_2_lvl[emo]}', dst+'.m4a')
                    src_audio = load_audio(src_audio)
                    dst_audio = load_audio(dst_audio)
                    frame_len = 16000 // self.fps
                    for i, fid in enumerate(path0):
                        src_clip = src_audio[fid*frame_len:(fid+1)*frame_len]
                        if src_clip.shape[0] != frame_len:
                            continue
                        batch = [src_clip]
                        for j in range(math.ceil(-self.nframe/2), math.ceil(self.nframe/2)):
                            k = np.clip(i+j, 0, len(path1)-1)
                            fid_ = path1[k]
                            dst_clip = dst_audio[fid_*frame_len:(fid_+1)*frame_len]
                            if dst_clip.shape[0] == frame_len:
                                batch.append(dst_clip)
                        if len(batch) == self.nframe + 1:
                            self.audio_clips.append(np.stack(batch, axis=0)[:, np.newaxis, :])
                            self.labels.append(label)
                            self.IDs.append(actor)
        
        LOG(f'{len(self.audio_clips)} samples loaded.')
    
    def __len__(self):
        return len(self.audio_clips)
    
    def __getitem__(self, index):
        batch = [self.audio_clips[index]]

        while len(batch) < self.bs:
            j = np.random.randint(0, len(self.audio_clips))
            if j == index or self.labels[j] != self.labels[index] or self.IDs[j] != self.IDs[index]:
                continue
            batch.append(self.audio_clips[j])

        return np.stack(batch, axis=0).astype(np.float32)


def LOG(*args, **kwargs):
    _LOG(time.strftime('%Y-%m-%d#%H:%M:%S',time.localtime(time.time())), end=' ')
    _LOG(*args, **kwargs)
    log_file.flush()


def _LOG(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_file)


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = AudioEncoder(args.depth, args.dim).to(device)
    mdist = MDist(args.loss_mode)
    loss_fn = Loss(args.loss_mode, device)
    dataset = AlignedDataset(args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, persistent_workers=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(args.epochs):
        for step, audios in enumerate(dataloader):
            _, bs, nframe, c, l = audios.shape
            audios = audios.view(bs*nframe, c, l).to(device)
            out = model(audios)
            out = out.view(bs, nframe, out.shape[-1])
            loss, _ = loss_fn(out)
            loss.backward()
            optimizer.step()

            mdist.update(out)

            if (step + 1) % args.log_interval == 0:
                LOG(f'epoch {epoch+1}, step [{step+1}/{len(dataloader)}], loss {loss.item():.4f}, '
                    f'dist ({mdist.pos_dist:.4f}, {mdist.neg_dist:.4f}), lr {optimizer.param_groups[0]["lr"]:.4g}')

        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(ckpt, os.path.join(args.work_dir, 'last.pth'))
    LOG('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth',type=int,default=14, help='model depth')
    parser.add_argument('--dim',type=int,default=128)
    parser.add_argument('--data_root',type=str,default='MEAD', help='data root')
    parser.add_argument('--aligned_path',type=str,default='MEAD/aligned_path.json', help='aligned_path.json')
    parser.add_argument('--actors', type=str, nargs='+', help='Subset of the MEAD actors', default=['M035', 'M023', 'W026', 'W025', 'W021', 'W019', 'M040', 'M011', 'W038', 'W023', 'W033', 'W040', 'M032', 'W036', 'M022', 'M039', 'W035', 'W016', 'M041', 'M027', 'M031', 'W014', 'M005', 'M019', 'M025', 'M042', 'M028', 'M037', 'M033', 'M024', 'W011', 'W028', 'W018', 'M034', 'M029', 'M007'])
    parser.add_argument('--emotions', type=str, nargs='+', help='Selection of Emotions', default=['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised'])
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--bs',type=int,default=64)
    parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--lr',type=float,default=0.0001)
    parser.add_argument('--wd',type=float,default=0., help='weight decay')
    parser.add_argument('--loss_mode', type=str, default='l2')
    parser.add_argument('--nframe',type=int,default=1, help='num frames for positive samples')
    parser.add_argument('--fps',type=int,default=30, help='video fps')
    parser.add_argument('--work_dir',type=str,default='exp/similarity/audio')
    parser.add_argument('--log_interval',type=int,default=50)
    parser.add_argument('--seed',type=int,default=2024)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.work_dir, exist_ok=True)
    log_file = open(os.path.join(args.work_dir, 'loss_log.txt'), 'w')

    main()

    log_file.close()
