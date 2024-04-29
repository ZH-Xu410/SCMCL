import argparse
import os
import json
import math
import time
import torch
import warnings
import pickle
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from models import ExprEncoder, Loss, MDist


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
        self.expr_clips = []
        self.labels = []
        self.IDs = []

        LOG('loading data...')
        for actor in self.actors:
            actor_root = os.path.join(
                self.data_root, '{}_deca_scmcl.pkl'.format(actor))
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
                    for i, fid in enumerate(path0):
                        if fid >= src_expr.shape[0]:
                            break
                        src_clip = src_expr[fid]
                        batch = [src_clip]
                        for j in range(math.ceil(-self.nframe/2), math.ceil(self.nframe/2)):
                            k = np.clip(i+j, 0, len(path1)-1)
                            fid_ = path1[k]
                            if fid_ < dst_expr.shape[0]:
                                dst_clip = dst_expr[fid_]
                                batch.append(dst_clip)
                        if len(batch) == self.nframe + 1:
                            self.expr_clips.append(np.stack(batch, axis=0)[:, np.newaxis, :])
                            self.labels.append(label)
                            self.IDs.append(actor)
        LOG(f'{len(self.expr_clips)} samples loaded.')

    def __len__(self):
        return len(self.expr_clips)
    
    def __getitem__(self, index):
        batch = [self.expr_clips[index]]

        while len(batch) < self.bs:
            j = np.random.randint(0, len(self.expr_clips))
            if j == index or self.labels[j] != self.labels[index] or self.IDs[j] != self.IDs[index]:
                continue
            batch.append(self.expr_clips[j])

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
    model = ExprEncoder(args.depth, args.dim).to(device)
    mdist = MDist(args.loss_mode)
    loss_fn = Loss(args.loss_mode, device)
    dataset = AlignedDataset(args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, persistent_workers=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    for epoch in range(args.epochs):
        for step, exprs in enumerate(dataloader):
            _, bs, nframe, c, l = exprs.shape
            exprs = exprs.view(bs*nframe, c, l).to(device)
            out = model(exprs)
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
            'optimizer': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.work_dir, 'last.pth'))
    LOG('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth',type=int,default=14, help='model depth')
    parser.add_argument('--dim',type=int,default=128)
    parser.add_argument('--data_root',type=str,default='MEAD/', help='data root')
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
    parser.add_argument('--work_dir',type=str,default='exp/similarity/3dmm')
    parser.add_argument('--log_interval',type=int,default=50)
    parser.add_argument('--seed',type=int,default=2024)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.work_dir, exist_ok=True)
    log_file = open(os.path.join(args.work_dir, 'loss_log.txt'), 'w')

    main()

    log_file.close()
