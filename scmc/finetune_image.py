import argparse
import os
import cv2
import json
import math
import pickle
import time
import torch
import warnings
import librosa
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader

from models import ImageEncoder, AudioEncoder, Loss, ScaleLayer, MDist
from cross_attention import CrossAttentionSeq


def load_audio(audio_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio, _ = librosa.load(audio_path, sr=16000)
        return audio


class AlignedDataset(Dataset):
    _emotions = ['neutral', 'angry', 'disgusted',
                 'fear', 'happy', 'sad', 'surprised']
    _levels = [1, 3, 3, 3, 3, 3, 3]
    _emo_2_lvl = {e: l for e, l in zip(_emotions, _levels)}

    def __init__(self, args):
        super().__init__()
        self.audio_root = args.audio_root
        self.image_root = args.image_root
        with open(args.aligned_path) as f:
            self.aligned_path = json.load(f)
        self.actors = args.actors
        self.emotions = args.emotions
        self.bs = args.bs
        self.nframe = args.nframe
        self.imsize = args.imsize
        self.fps = args.fps

        self.img_paths = []
        self.audio_clips = []
        self.labels = []
        self.IDs = []

        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3, 1, 1))
        self.margin = args.margin // 2

        cache_path = '.cache/finetune_image_tmpr_dataset.pkl'
        if os.path.exists(cache_path):
            LOG('load data from cache')
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            self.img_paths = cache_data['img_paths']
            self.audio_clips = cache_data['audio_clips']
            self.labels = cache_data['labels']
            self.IDs = cache_data['IDs']
            LOG(f'{len(self.img_paths)} samples loaded.')
        else:
            self.load_data()
            cache_data = {
                'img_paths': self.img_paths,
                'audio_clips': self.audio_clips,
                'labels': self.labels,
                'IDs': self.IDs
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
    
    def load_data(self):
        LOG('loading data...')
        for actor in self.actors:
            with open(os.path.join(self.image_root, actor, 'videos/_frame_info.txt')) as f:
                frame_infos = f.read().splitlines()
            id_map = dict([x.split(' ') for x in frame_infos])
            img_dir = os.path.join(self.image_root, actor, 'faces_aligned')

            for label, emo in enumerate(self.emotions):
                if emo == 'neutral':
                    continue

                for k, v in self.aligned_path[actor][emo].items():
                    src, dst = k.split('_')
                    path0, path1 = v
                    src_audio = os.path.join(
                        self.audio_root, actor, 'audio/neutral/level_1', src+'.m4a')
                    dst_audio = os.path.join(
                        self.audio_root, actor, 'audio', emo, f'level_{self._emo_2_lvl[emo]}', dst+'.m4a')
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

                        img_src_batch = []
                        img_dst_batch = []
                        for j in range(self.nframe):
                            if f'neutral_{src}_{fid0+j}' not in id_map or f'{emo}_{dst}_{fid1+j}' not in id_map:
                                break
                            global_id = int(id_map[f'neutral_{src}_{fid0+j}'])
                            img_src_batch.append(os.path.join(
                                img_dir, f'{global_id//50:06d}/{global_id:06d}.png'))
                            global_id = int(id_map[f'{emo}_{dst}_{fid1+j}'])
                            img_dst_batch.append(os.path.join(
                                img_dir, f'{global_id//50:06d}/{global_id:06d}.png'))

                        all_exits = True
                        img_batch = img_src_batch + img_dst_batch
                        for x in img_batch:
                            if not os.path.exists(x):
                                # print(f'warning: image not exists {x}')
                                all_exits = False
                                break
                        if all_exits and len(img_batch) == self.nframe * 2:
                            self.audio_clips.append(
                                np.stack(audio_src_batch+audio_dst_batch, axis=0)[:, np.newaxis, :])
                            self.img_paths.append(img_batch)
                            self.labels.append(label)
                            self.IDs.append(actor)
        LOG(f'{len(self.img_paths)} samples loaded.')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_batch = [self.img_paths[index]]
        audio_batch = [self.audio_clips[index]]

        while len(img_batch) < self.bs:
            j = np.random.randint(0, len(self.img_paths))
            if j == index or self.labels[j] != self.labels[index] or self.IDs[j] != self.IDs[index]:
                continue
            img_batch.append(self.img_paths[j])
            audio_batch.append(self.audio_clips[j])

        batch_data = []
        for x in img_batch:
            data = []
            for img in x:
                img = cv2.imread(img)[:, :, ::-1]
                h, w, _ = img.shape
                img = img[self.margin:h-self.margin,
                          self.margin:w-self.margin, :]
                img = cv2.resize(img, (self.imsize, self.imsize),
                                 interpolation=cv2.INTER_LINEAR)
                data.append(img)
            batch_data.append(np.stack(data, axis=0).transpose((0, 3, 1, 2)))
        batch_data = np.stack(batch_data, axis=0)
        batch_data = (batch_data/255. - self.mean) / self.std
        return batch_data.astype(np.float32),  np.stack(audio_batch, axis=0).astype(np.float32)


def LOG(*args, **kwargs):
    _LOG(time.strftime('%Y-%m-%d#%H:%M:%S', time.localtime(time.time())), end=' ')
    _LOG(*args, **kwargs)
    log_file.flush()


def _LOG(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_file)


def main():
    assert args.nframe > 1
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    image_encoder = ImageEncoder(args.depth1, args.dim).to(device).train()
    # image_encoder.model.fc.train()
    image_ckpt = torch.load(args.image_ckpt, device)['model']
    image_encoder.load_state_dict(image_ckpt)
    audio_encoder = AudioEncoder(args.depth2, args.dim).to(device).train()
    # audio_encoder.fc.train()
    audio_ckpt = torch.load(args.audio_ckpt, device)['model']
    audio_encoder.load_state_dict(audio_ckpt)
    image_attn = CrossAttentionSeq(
        num_blocks=args.num_blocks,
        in_features=image_encoder.feat_dim,
        dim=args.dim, heads=4).to(device).train()
    audio_attn = CrossAttentionSeq(
        num_blocks=args.num_blocks,
        in_features=audio_encoder.feat_dim,
        dim=args.dim, heads=4).to(device).train()
    scale_layer = ScaleLayer().to(device).train()
    loss_fn = Loss(args.loss_mode, device)
    audio_mdist = MDist(args.loss_mode)
    image_mdist = MDist(args.loss_mode)
    dataset = AlignedDataset(args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                            num_workers=args.num_workers, persistent_workers=True)
    optimizer = torch.optim.AdamW(
        list(audio_encoder.parameters()) +
        list(image_encoder.parameters()) +
        list(scale_layer.parameters()) +
        list(image_attn.parameters()) +
        list(audio_attn.parameters()),
        lr=args.lr, weight_decay=args.wd)

    for epoch in range(args.epochs):
        for step, (imgs, audios) in enumerate(dataloader):
            _, bs, nframe, c, h, w = imgs.shape
            imgs = imgs.view(bs*nframe, c, h, w).to(device)
            _, bs, nframe, c, l = audios.shape
            audios = audios.view(bs*nframe, c, l).to(device)
            image_vec = image_encoder.encode(imgs)
            image_vec = image_vec.view(bs, nframe, *image_vec.shape[1:])
            fused_image_vec = [image_vec[:, 0], image_vec[:, nframe//2]]
            for i in range(1, args.nframe):
                fused_image_vec[0] = image_attn(
                    image_vec[:, i], fused_image_vec[0])
                fused_image_vec[1] = image_attn(
                    image_vec[:, nframe//2+i], fused_image_vec[1])
            fused_image_vec[0] = image_encoder.activate(fused_image_vec[0])
            fused_image_vec[1] = image_encoder.activate(fused_image_vec[1])
            fused_image_vec = torch.stack(fused_image_vec, dim=1)
            loss_image, image_dist = loss_fn(fused_image_vec)
            if step > 5000:
                image_dist = scale_layer(image_dist)

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
                torch.abs(image_dist - audio_dist).mean() if step > 5000 else audio_dist.new_zeros(1)
            loss = loss_audio + loss_image + loss_consis
            loss.backward()
            optimizer.step()

            image_mdist.update(fused_image_vec, scale_layer)
            audio_mdist.update(fused_audio_vec)

            if (step + 1) % args.log_interval == 0:
                LOG(f'epoch {epoch+1}, step [{step+1}/{len(dataloader)}], loss_audio {loss_audio.item():.4f}, loss_image {loss_image.item():.4f}, '
                    f'loss_consis {loss_consis.item():.4f}, ', 
                    f'image_dist ({image_mdist.pos_dist:.4f}, {image_mdist.neg_dist:.4f}), '
                    f'audio_dist ({audio_mdist.pos_dist:.4f}, {audio_mdist.neg_dist:.4f})')

        ckpt = {
            'model': {
                'audio_encoder': audio_encoder.state_dict(),
                'image_encoder': image_encoder.state_dict(),
                'scale_layer': scale_layer.state_dict(),
                'image_attn': image_attn.state_dict(),
                'audio_attn': audio_attn.state_dict()
            },
            'optimizer': optimizer.state_dict()
        }
        torch.save(ckpt, os.path.join(args.work_dir, 'last.pth'))
    LOG('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth1', type=int, default=18,
                        help='image encoder depth')
    parser.add_argument('--depth2', type=int, default=14,
                        help='audio encoder depth')
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--num_blocks', type=int, default=1,
                        help='num blocks for cross attention')
    parser.add_argument('--image_ckpt', type=str,
                        default='exp/similarity/image/last.pth', help='image encoder checkpoint path')
    parser.add_argument('--audio_ckpt', type=str,
                        default='exp/similarity/audio/last.pth', help='audio encoder checkpoint path')
    parser.add_argument('--image_root', type=str,
                        default='celeb/train', help='data root')
    parser.add_argument('--audio_root', type=str,
                        default='MEAD', help='data root')
    parser.add_argument('--aligned_path', type=str,
                        default='MEAD/aligned_path.json', help='aligned_path.json')
    parser.add_argument('--actors', type=str, nargs='+', help='Subset of the MEAD actors',
                        default=['M035', 'M023', 'W026', 'W025', 'W021', 'W019', 'M040', 'M011', 'W038', 'W023', 'W033', 'W040', 'M032', 'W036', 'M022', 'M039', 'W035', 'W016', 'M041', 'M027', 'M031', 'W014', 'M005', 'M019', 'M025', 'M042', 'M028', 'M037', 'M033', 'M024', 'W011', 'W028', 'W018', 'M034', 'M029', 'M007'])
    parser.add_argument('--emotions', type=str, nargs='+', help='Selection of Emotions',
                        default=['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised'])
    parser.add_argument('--imsize', type=int, default=64, help='image size')
    parser.add_argument('--margin', type=int, default=70)
    parser.add_argument('--fps', type=int, default=30, help='video fps')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--wd', type=float, default=0., help='weight decay')
    parser.add_argument('--loss_mode', type=str, default='l2')
    parser.add_argument('--nframe', type=int, default=2,
                        help='num frames for positive samples')
    parser.add_argument('--lambd_consis', type=float, default=1,
                        help='consistency loss weight')
    parser.add_argument('--work_dir', type=str,
                        default='exp/similarity/image_finetune')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.work_dir, exist_ok=True)
    log_file = open(os.path.join(args.work_dir, 'loss_log.txt'), 'w')

    main()

    log_file.close()
