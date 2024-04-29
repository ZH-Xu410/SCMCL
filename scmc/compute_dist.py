import argparse
import torch
import warnings
import os
import pickle
import librosa
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from similarity.models import AudioEncoder
from similarity.cross_attention import CrossAttentionSeq


emotions = ['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
levels = [1, 3, 3, 3, 3, 3, 3]
emo_2_lvl = {e: l for e, l in zip(emotions, levels)}


def load_audio(audio_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio, _ = librosa.load(audio_path, sr=16000)
        return audio


def load_data(args):
    print('loading data...')
    data = OrderedDict()
    frame_len = 16000 // args.fps
    for actor in args.actors:
        actor_root = os.path.join(
            args.root, '{}_deca_avcl.pkl'.format(actor))
        actor_data = pickle.load(open(actor_root, "rb"))
        audio_clips = {}
        for emo in args.emotions:
            audio_clips[emo] = []
            for name in actor_data[emo]:
                audio_path = os.path.join(
                    args.root, actor, 'audio', emo, f'level_{emo_2_lvl[emo]}', name+'.m4a')
                if not os.path.exists(audio_path):
                    continue

                tgt_audio = load_audio(audio_path)
                for i in range(0, actor_data[emo][name].shape[0]-args.seq_len, args.hop_len):
                    j = i + args.seq_len - 1  # last frame
                    clip = tgt_audio[(j-1)*frame_len:(j+1)*frame_len]
                    if len(clip) == 2*frame_len:
                        audio_clips[emo].append(clip)

        data[actor] = audio_clips
    print('done.')
    return data


class TestDataset(Dataset):
    def __init__(self, audio_clips):
        super().__init__()
        self.audio_clips = audio_clips

    def __len__(self):
        return len(self.audio_clips)

    def __getitem__(self, index):
        return self.audio_clips[index][np.newaxis, :]


def get_dataloader(audio_clips, batch_size):
    dataset = TestDataset(audio_clips)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            drop_last=False,
                            pin_memory=True)
    return dataloader


@torch.no_grad()
def compute_dist(actor, tgt_vecs, device):
    src_dir = os.path.join(args.save_dir, actor, 'neutral')
    dists = []
    for src_vecs in sorted(os.listdir(src_dir)):
        src_vecs = torch.from_numpy(np.load(os.path.join(src_dir, src_vecs))).to(device)
        dists.append(torch.cdist(src_vecs, tgt_vecs))
    return torch.cat(dists, dim=0).cpu().numpy()


def main(args):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    encoder = AudioEncoder(args.depth, args.dim).to(device)
    attn = CrossAttentionSeq(encoder.feat_dim, args.dim, args.num_blocks, heads=4).to(device)
    ckpt = torch.load(args.ckpt, device)['model']
    encoder.load_state_dict(ckpt['audio_encoder'])
    attn.load_state_dict(ckpt['audio_attn'])
    encoder.eval()
    attn.eval()
    data = load_data(args)
    dists = {}

    with torch.no_grad():
        for actor in data:
            dists[actor] = {}
            for emo, audio_clips in data[actor].items():
                dists[actor][emo] = []
                outputs = []
                index = 0
                save_dir = os.path.join(args.save_dir, actor, emo)
                os.makedirs(save_dir, exist_ok=True)
                dataloader = get_dataloader(audio_clips, args.bs)
                for i, batch in enumerate(tqdm(dataloader, desc=f'{actor} {emo} {len(audio_clips)} audio clips')):
                    batch1, batch2 = batch.chunk(2, dim=-1)
                    vec1 = encoder.encode(batch1.to(device))
                    vec2 = encoder.encode(batch2.to(device))
                    fused_vec = encoder.activate(attn(vec2, vec1))
                    outputs.append(fused_vec)
                    if len(outputs) * args.bs >= args.lbs or i + 1 == len(dataloader):
                        out = torch.cat(outputs, dim=0)
                        np.save(os.path.join(save_dir, f'vectors_{index:04d}.npy'), out.cpu().numpy())
                        outputs = []
                        index += 1

                        if emo != 'neutral':
                            dists[actor][emo].append(compute_dist(actor, out, device))

                if len(dists[actor][emo]):
                    dists[actor][emo] = np.concatenate(dists[actor][emo], axis=1)

    with open(os.path.join(args.save_dir, 'dists.pkl'), 'wb') as f:
        pickle.dump(dists, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=14,
                        help='audio encoder depth')
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--num_blocks', type=int, default=1,
                        help='num blocks for cross attention')
    parser.add_argument(
        '--ckpt', type=str, default='exp/similarity/3dmm_finetune/last.pth', help='audio checkpoint path')
    parser.add_argument('--root', type=str,
                        default='MEAD', help='data root')
    parser.add_argument('--actors', type=str, nargs='+',
                        help='Subset of the MEAD actors', default=['M003', 'M009', 'W029', 'M012', 'M030', 'W015'])
    parser.add_argument('--emotions', type=str, nargs='+',
                        help='Selection of Emotions', default=['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised'])
    parser.add_argument('--seq_len', type=int, default=10,
                        help='Length of exp. coeffs. sequence')
    parser.add_argument('--hop_len', type=int, default=1,
                        help='Hop Length (set to 1 by default for test)')
    parser.add_argument('--fps', type=int, default=30, help='video fps')
    parser.add_argument('--bs', type=int, default=256,
                        help='batch size for model inference')
    parser.add_argument('--lbs', type=int, default=4096,
                        help='larger batch size for distance calculation')
    parser.add_argument('--save_dir', type=str,
                        default='exp/similarity/dists_for_3dmm')
    args = parser.parse_args()

    main(args)
