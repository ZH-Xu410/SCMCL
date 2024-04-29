import argparse
import os
import cv2
import json
import math
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import ImageEncoder, Loss, MDist


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
        self.imsize = args.imsize
        self.tau = args.tau
        self.img_paths = []
        self.labels = []
        self.IDs = []

        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3, 1, 1))
        self.margin = args.margin // 2

        LOG('loading data...')
        for actor in self.actors:
            with open(os.path.join(self.data_root, actor, 'videos/_frame_info.txt')) as f:
                frame_infos = f.read().splitlines()
            id_map = dict([x.split(' ') for x in frame_infos])
            img_dir = os.path.join(self.data_root, actor, 'faces_aligned')
            for label, emo in enumerate(self.emotions):
                if emo == 'neutral':
                    continue
                for k, v in self.aligned_path[actor][emo].items():
                    src, dst = k.split('_')
                    path0, path1 = v
                    for i, fid in enumerate(path0):
                        if f'neutral_{src}_{fid}' not in id_map:
                            continue
                        global_id = int(id_map[f'neutral_{src}_{fid}'])
                        batch = [os.path.join(img_dir, f'{global_id//50:06d}/{global_id:06d}.png')]
                        for j in range(math.ceil(-self.nframe/2), math.ceil(self.nframe/2)):
                            k = np.clip(i+j, 0, len(path1)-1)
                            if f'{emo}_{dst}_{path1[k]}' not in id_map:
                                continue
                            global_id = int(id_map[f'{emo}_{dst}_{path1[k]}'])
                            batch.append(os.path.join(img_dir, f'{global_id//50:06d}/{global_id:06d}.png'))
                        all_exits = True
                        for x in batch:
                            if not os.path.exists(x):
                                # print(f'warning: image not exists {x}')
                                all_exits = False
                                break
                        if all_exits and len(batch) == self.nframe + 1:
                            self.img_paths.append(batch)
                            self.labels.append(label)
                            self.IDs.append(actor)
        LOG(f'{len(self.img_paths)} samples loaded.')
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        batch = [self.img_paths[index]]

        while len(batch) < self.bs:
            j = np.random.randint(0, len(self.img_paths))
            if abs(j - index) <= self.tau or self.labels[j] != self.labels[index] or self.IDs[j] != self.IDs[index]:
                continue
            batch.append(self.img_paths[j])

        batch_data = []
        for x in batch:
            data = []
            for img in x:
                img = cv2.imread(img)[:, :, ::-1]
                h, w, _ = img.shape
                img = img[self.margin:h-self.margin, self.margin:w-self.margin, :]
                img = cv2.resize(img, (self.imsize, self.imsize), interpolation=cv2.INTER_LINEAR)
                data.append(img)
            batch_data.append(np.stack(data, axis=0).transpose((0, 3, 1, 2)))
        batch_data = np.stack(batch_data, axis=0)
        batch_data = (batch_data/255. - self.mean) / self.std
        return batch_data.astype(np.float32)


def LOG(*args, **kwargs):
    _LOG(time.strftime('%Y-%m-%d#%H:%M:%S',time.localtime(time.time())), end=' ')
    _LOG(*args, **kwargs)
    log_file.flush()


def _LOG(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_file)


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.makedirs(args.work_dir, exist_ok=True)
    model = ImageEncoder(args.depth, args.dim).to(device)
    mdist = MDist(args.loss_mode)
    loss_fn = Loss(args.loss_mode, device)
    dataset = AlignedDataset(args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, persistent_workers=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(args.epochs):
        for step, imgs in enumerate(dataloader):
            _, bs, nframe, c, h, w = imgs.shape
            imgs = imgs.view(bs*nframe, c, h, w).to(device)
            out = model(imgs)
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
    parser.add_argument('--depth',type=int,default=18, help='model depth')
    parser.add_argument('--dim',type=int,default=128)
    parser.add_argument('--data_root',type=str,default='MEAD/', help='data root')
    parser.add_argument('--aligned_path',type=str,default='MEAD/aligned_path.json', help='aligned_path.json')
    parser.add_argument('--actors', type=str, nargs='+', help='Subset of the MEAD actors', default=['M035', 'M023', 'W026', 'W025', 'W021', 'W019', 'M040', 'M011', 'W038', 'W023', 'W033', 'W040', 'M032', 'W036', 'M022', 'M039', 'W035', 'W016', 'M041', 'M027', 'M031', 'W014', 'M005', 'M019', 'M025', 'M042', 'M028', 'M037', 'M033', 'M024', 'W011', 'W028', 'W018', 'M034', 'M029', 'M007'])
    parser.add_argument('--emotions', type=str, nargs='+', help='Selection of Emotions', default=['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised'])
    parser.add_argument('--imsize',type=int,default=128, help='image size')
    parser.add_argument('--margin', type=int, default=70)
    parser.add_argument('--tau', type=int, default=3)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--bs',type=int,default=64)
    parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--lr',type=float,default=0.0001)
    parser.add_argument('--wd',type=float,default=0., help='weight decay')
    parser.add_argument('--loss_mode', type=str, default='l2')
    parser.add_argument('--nframe',type=int,default=1, help='num frames for positive samples')
    parser.add_argument('--work_dir',type=str,default='exp/similarity/image')
    parser.add_argument('--log_interval',type=int,default=50)
    parser.add_argument('--seed',type=int,default=2024)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.work_dir, exist_ok=True)
    log_file = open(os.path.join(args.work_dir, 'loss_log.txt'), 'w')

    main()

    log_file.close()
