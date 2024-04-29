import json
import warnings
import os
import pickle
import numpy as np
import torch
import librosa
from munch import Munch
from torch.utils import data


audio_postfix = '.m4a' # '.wav'

def load_audio(audio_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio, _ = librosa.load(audio_path, sr=16000)
        return audio


class SCMCL_MEAD(data.Dataset):
    """Dataset class for the audio-visual correlated MEAD dataset."""

    emotions = ['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
    levels = [1, 3, 3, 3, 3, 3, 3]
    emo_2_lvl = {e:l for e, l in zip(emotions, levels)}
    topk = 10
    fps = 30
    _prefix = 'MEAD-6ID'

    def __init__(self, opt, which='source', phase='train', return_img_info=False, cross_exp_pseudo=True, prefix=None):
        super().__init__()
        prefix = prefix or self._prefix
        self.cache_file = f'.cache/SCMCL_{prefix}-dataset-{which}-{phase}.pkl'
        self.opt = opt
        self.which = which
        self.phase = phase
        self.seq_len = opt.seq_len
        self.hop_len = opt.hop_len
        self.root = opt.train_root
        self.deca_root = opt.deca_root
        self.return_img_info = return_img_info
        self.cross_exp_pseudo = cross_exp_pseudo

        with open(opt.dist_file, 'rb') as f:
            self.dists = pickle.load(f)
        self.dist_thresh = opt.dist_thresh
        if phase == 'train':
            self.selected_actors = opt.selected_actors
        elif phase == 'val':
            self.selected_actors = opt.selected_actors_val
        self.selected_emotions = opt.selected_emotions
        
        self.selected_labels = [self.emotions.index(
            e) for e in self.selected_emotions]

        self.seqs = []
        self.pseudos = []
        self.labels = []
        if not self.check_cache():
           self.load_data()

        self.num_seqs = len(self.seqs)

        if self.which == 'reference':
            inds = np.random.permutation(self.num_seqs)
            self.seqs = [self.seqs[i] for i in inds]
            self.labels = [self.labels[i] for i in inds]
    
    def check_cache(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        if not os.path.exists(self.cache_file):
            return False
        with open(self.cache_file,'rb') as f:
            cache = pickle.load(f)
        check_field = cache['check_field']
        s = json.dumps({
            'root': self.root,
            'actors': self.selected_actors,
            'emotions': self.selected_emotions,
            'seq_len': self.seq_len, 
            'hop_len': self.hop_len,
            'dist_thresh': self.dist_thresh
        })
        if check_field == s:
            self.seqs = cache['seqs']
            self.pseudos = cache['pseudos']
            self.labels = cache['labels']
            print('load dataset from cache.')
            return True
        else:
            return False
    
    def load_data(self):
        if self.which != 'reference' and self.phase == 'train':
            self.prepare_pseudos()

        print('loading dataset...')
        frame_len = 16000 // self.fps
        for actor in self.selected_actors:
            actor_root = os.path.join(
                self.deca_root, '{}_deca_scmcl.pkl'.format(actor))
            assert os.path.isfile(
                actor_root), '%s is not a valid file' % actor_root

            actor_data = pickle.load(open(actor_root, "rb"))
            for emo in self.selected_emotions:
                index = 0
                for name in actor_data[emo]:
                    audio_path = os.path.join(self.root, actor, 'audio', emo, f'level_{self.emo_2_lvl[emo]}', name+audio_postfix)
                    if not os.path.exists(audio_path):
                        continue
                    params = actor_data[emo][name]
                    params = np.concatenate((params[:, 0:1], params[:, 3:]), 1) # 51
                    audio_len = len(load_audio(audio_path))

                    for i in range(0, params.shape[0]-self.seq_len, self.hop_len):
                        frame_id = i + self.seq_len - 1
                        if (frame_id+1)*frame_len > audio_len:
                            break
                        img_info = {'actor': actor, 'emotion': emo, 'video_name': name, 'frame_id': frame_id}
                        self.seqs.append([params[i:i + self.seq_len], img_info])
                        self.labels.append(self.emotions.index(emo))
                        if self.which == 'reference' or self.phase != 'train':
                            continue

                        self.pseudos.append([None] * len(self.emotions))
                        for label, emo_ in enumerate(self.emotions):
                            if emo_ == emo:
                                self.pseudos[-1][label] = [self.seqs[-1]+[0.]]
                                continue
                            
                            if emo == 'neutral':
                                #if index >= self.dists[actor][emo_].shape[0]:
                                #    continue
                                dist = self.dists[actor][emo_][index]
                                inds = np.arange(dist.shape[0])[dist <= self.dist_thresh]
                                if len(inds):
                                    inds_ = np.argsort(dist[inds])
                                    self.pseudos[-1][label] = [self._seqs[actor][emo_][j] + [dist[j]] for j in inds[inds_]]
                                    if len(self.pseudos[-1][label]) > self.topk:
                                        self.pseudos[-1][label] = self.pseudos[-1][label][:self.topk]
                            elif emo_ == 'neutral':
                                #if index >= self.dists[actor][emo].shape[1]:
                                #    continue
                                dist = self.dists[actor][emo][:, index]
                                inds = np.arange(dist.shape[0])[dist <= self.dist_thresh]
                                if len(inds):
                                    inds_ = np.argsort(dist[inds])
                                    self.pseudos[-1][label] = [self._seqs[actor][emo_][j] + [dist[j]] for j in inds[inds_]]
                                    if len(self.pseudos[-1][label]) > self.topk:
                                        self.pseudos[-1][label] = self.pseudos[-1][label][:self.topk]
                            elif self.cross_exp_pseudo:
                                #if index >= self.dists[actor][emo].shape[0]:
                                #    continue
                                dist_e1 = self.dists[actor][emo][:, index]
                                inds = np.arange(dist_e1.shape[0])[dist_e1 <= self.dist_thresh]
                                if len(inds):
                                    j = inds[np.argmin(dist_e1[inds])]
                                    dist_e2 = self.dists[actor][emo_][j]
                                    inds = np.arange(dist_e2.shape[0])[dist_e2 <= self.dist_thresh]
                                    if len(inds):
                                        inds_ = np.argsort(dist_e2[inds])[:1]
                                        self.pseudos[-1][label] = [self._seqs[actor][emo_][k] + [dist_e2[k]] + self._seqs[actor]['neutral'][j] + [dist_e1[j]] for k in inds[inds_]]

                        index += 1

        s = json.dumps({
            'root': self.root,
            'actors': self.selected_actors,
            'emotions': self.selected_emotions,
            'seq_len': self.seq_len, 
            'hop_len': self.hop_len,
            'dist_thresh': self.dist_thresh
        })
        cache_data = {
            'check_field': s, 
            'seqs': self.seqs, 
            'pseudos': self.pseudos, 
            'labels': self.labels}
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f'{len(self.seqs)} samples loaded.')
    
    def __getitem__(self, index):
        """Return one sequence and its corresponding label."""
        param, img_info = self.seqs[index]
        param = torch.FloatTensor(param)
        label = self.labels[index]
        if self.return_img_info:
            return index, param, label, img_info
        else:
            return index, param, label

    def __len__(self):
        """Return the number of sequences."""
        return len(self.seqs)
    
    def prepare_pseudos(self):
        print('preparing pseudos...')

        self._seqs = {}
        frame_len = 16000 // self.fps
        for actor in self.selected_actors:
            actor_root = os.path.join(
                self.deca_root, '{}_deca_scmcl.pkl'.format(actor))
            assert os.path.isfile(
                actor_root), '%s is not a valid file' % actor_root

            actor_data = pickle.load(open(actor_root, "rb"))

            self._seqs[actor] = {}
            for emo in self.emotions:
                # if emo == 'neutral':
                #     continue
                self._seqs[actor][emo] = []
                for name in actor_data[emo]:
                    audio_path = os.path.join(self.root, actor, 'audio', emo, f'level_{self.emo_2_lvl[emo]}', name+audio_postfix)
                    if not os.path.exists(audio_path):
                        continue
                    params = actor_data[emo][name]
                    params = np.concatenate((params[:, 0:1], params[:, 3:]), 1) # 51
                    audio_len = len(load_audio(audio_path))

                    for i in range(0, params.shape[0]-self.seq_len, self.hop_len):
                        if (i+1)*frame_len > audio_len:
                            break
                        frame_id = i + self.seq_len - 1
                        img_info = {'actor': actor, 'emotion': emo, 'video_name': name, 'frame_id': frame_id}
                        self._seqs[actor][emo].append([params[i:i + self.seq_len], img_info])

    def get_pseudo(self, index, label):
        if self.labels[index] != 0 and label != 0 and not self.cross_exp_pseudo:
            return None
        pseudo_list = self.pseudos[index][label]
        if pseudo_list is None or len(pseudo_list) == 0:
            return None
        else:
            choice = np.random.randint(0, len(pseudo_list), 1)
            pseudo = pseudo_list[int(choice)]
            if len(pseudo) == 3:
                param, img_info, dist = pseudo
                inter_param, inter_info, inter_dist = np.zeros_like(param), None, 0
            else:
                param, img_info, dist, inter_param, inter_info, inter_dist = pseudo
            if self.return_img_info:
                return torch.FloatTensor(param), dist, img_info, torch.FloatTensor(inter_param), inter_dist, inter_info
            else:
                return torch.FloatTensor(param), dist, torch.FloatTensor(inter_param), inter_dist


def get_train_loader(opt, which, prefix=None):
    dataset = SCMCL_MEAD(opt, which, prefix=prefix)
    return data.DataLoader(dataset=dataset,
                           batch_size=opt.batch_size,
                           shuffle=True,
                           num_workers=opt.nThreads,
                           drop_last=True,
                           pin_memory=True)


def get_val_loader(opt, which, prefix=None):
    dataset = SCMCL_MEAD(opt, which, phase='val', prefix=prefix)
    return data.DataLoader(dataset=dataset,
                           batch_size=opt.batch_size,
                           shuffle=False,
                           num_workers=opt.nThreads,
                           drop_last=True,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader, loader_ref, pseudo=True):
        self.loader = loader
        self.loader_ref = loader_ref
        self.pseudo = pseudo

        self.iter = iter(self.loader)
        self.iter_ref = iter(self.loader_ref)

    def _fetch_inputs(self, label):
        try:
            ids, x, y = next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            ids, x, y = next(self.iter)
        x_tgt = []
        inter_tgt = []
        dists = []
        inter_dists = []
        tgt_mask = []
        for i, l in zip(ids, label):
            pseudo = self.loader.dataset.get_pseudo(i, int(l)) \
                if self.pseudo and self.loader.dataset.phase == 'train' else None
            if pseudo is None:
                x_tgt_ = np.zeros_like(x[0])
                dist = np.inf
                tgt_mask.append(0)
                inter_tgt_ = np.zeros_like(x[0])
                inter_dist = 0
            else:
                x_tgt_, dist, inter_tgt_, inter_dist = pseudo
                tgt_mask.append(1)
            x_tgt.append(x_tgt_)
            dists.append(dist)
            inter_tgt.append(inter_tgt_)
            inter_dists.append(inter_dist)
        x_tgt = torch.from_numpy(np.stack(x_tgt, axis=0))
        tgt_mask = torch.from_numpy(np.array(tgt_mask, dtype=np.float32))
        dists = torch.from_numpy(np.array(dists, dtype=np.float32))
        inter_tgt = torch.from_numpy(np.stack(inter_tgt, axis=0))
        inter_dists = torch.from_numpy(np.array(inter_dists, dtype=np.float32))
        return x, y, x_tgt, tgt_mask, dists, inter_tgt, inter_dists

    def _fetch_refs(self):
        try:
            _, x, y = next(self.iter_ref)
        except StopIteration:
            self.iter_ref = iter(self.loader_ref)
            _, x, y = next(self.iter_ref)
        return x, y

    def __next__(self):
        x_ref, y_ref = self._fetch_refs()
        x, y, x_tgt, tgt_mask, dists, inter_tgt, inter_dists = self._fetch_inputs(y_ref)
        inputs = Munch(x_src=x, y_src=y, x_ref=x_ref,
                       y_ref=y_ref, x_tgt=x_tgt, 
                       tgt_mask=tgt_mask, dists=dists,
                       inter_tgt=inter_tgt, inter_dists=inter_dists)

        return inputs


if __name__ == '__main__':
    opt = Munch(
        train_root='MEAD',
        seq_len=10,
        hop_len=1,
        selected_actors=['M003', 'M009', 'W029'],
        selected_emotions=['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised'],
        dist_file='exp/similarity/dists/dists.pkl',
        dist_thresh=1.0
    )

    dataset = SCMCL_MEAD(opt)
    print(len(dataset))
    index, param, label = dataset[3010]
    print(label, param.shape)

    neutral_pseudo = dataset.get_pseudo(index, 0)
    print(neutral_pseudo[0].shape if neutral_pseudo is not None else 'neutral pseudo is None')
    happy_pseudo = dataset.get_pseudo(index, 1)
    print(happy_pseudo[0].shape if happy_pseudo is not None else 'happy pseudo is None')
    surp_pseudo = dataset.get_pseudo(index, 6)
    print(surp_pseudo[0].shape if surp_pseudo is not None else 'surprised pseudo is None')
