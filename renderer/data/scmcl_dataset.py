import os
import torch
import numpy as np
from copy import deepcopy
from PIL import Image
from munch import Munch
from torch.utils import data
from manipulator.data.scmcl_dataset import SCMCL_MEAD
from renderer.data.video_dataset import videoDataset
from renderer.data.base_dataset import get_params, get_transform, get_video_parameters
from renderer.data.landmarks_to_image import create_eyes_image


class SCMCLDataset(data.Dataset):
    emotions = ['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
    levels = [1, 3, 3, 3, 3, 3, 3]
    emo_2_lvl = {e:l for e, l in zip(emotions, levels)}
    sample_freq = 1
    _prefix = '6ID'

    def __init__(self, opt, which='source', phase='train', prefix=None):
        super().__init__()
        self.opt = opt
        prefix = prefix or self._prefix
        self.mdataset = SCMCL_MEAD(opt, which, phase, return_img_info=True, prefix='-'.join(opt.selected_actors+[prefix]))
        self.rdataset = {}
        self.frame_infos = {}
        for actor in opt.selected_actors:
            opt_ = deepcopy(opt)
            opt_.celeb = os.path.join(opt.celeb, actor)
            d = videoDataset()
            d.initialize(opt_)
            self.rdataset[actor] = d
            with open(os.path.join(opt.celeb, actor, 'videos/_frame_info.txt'), 'r') as f:
                infos = f.read().splitlines()
            self.frame_infos[actor] = dict([tuple(x.split(' ')) for x in infos])

    def __len__(self):
        return len(self.mdataset) // self.sample_freq
    
    def get_img_path(self, img_info):
        if img_info is None:
            return None
        actor = img_info['actor']
        emo = img_info['emotion']
        name = img_info['video_name']
        frame_id = img_info['frame_id']
        paths = []
        for i in range(self.opt.n_frames_G):
            if f'{emo}_{name}_{frame_id-i}' not in self.frame_infos[actor]:
                return None
            global_idx = int(self.frame_infos[actor][f'{emo}_{name}_{frame_id-i}'])
            seq_idx = global_idx // 50
            img_path = os.path.join(self.opt.celeb, actor, 'images', f'{seq_idx:06d}', f'{global_idx:06d}.png')
            if not os.path.exists(img_path):
                return None
            paths.insert(0, img_path)
        return paths

    def _get_render_data(self, img_info):
        actor = img_info['actor']
        emo = img_info['emotion']
        name = img_info['video_name']
        frame_id = img_info['frame_id']
        if f'{emo}_{name}_{frame_id}' not in self.frame_infos[actor]:
            return None
        global_idx = int(self.frame_infos[actor][f'{emo}_{name}_{frame_id}'])
        seq_idx = global_idx // 50
        local_idx = global_idx % 50
        img_path = os.path.join(self.opt.celeb, actor, 'images', f'{seq_idx:06d}', f'{global_idx:06d}.png')
        if not os.path.exists(img_path):
            return None
        d = self.rdataset[actor]
        nmfc_video_paths = d.nmfc_video_paths[seq_idx]
        rgb_video_paths = d.rgb_video_paths[seq_idx]
        if self.opt.use_shapes:
            shape_video_paths = d.shape_video_paths[seq_idx]
        landmark_video_paths = d.landmark_video_paths[seq_idx]
        mask_video_paths = d.mask_video_paths[seq_idx]

        # Get parameters and transforms.
        first_nmfc_image = Image.open(
            nmfc_video_paths[local_idx-self.opt.n_frames_G+1]).convert('RGB')
        params = get_params(self.opt, first_nmfc_image.size)
        transform_scale_nmfc_video = get_transform(self.opt, params, normalize=False,
                                                   augment=not self.opt.no_augment_input and self.opt.isTrain)  # do not normalize nmfc but augment.
        # get_transform(self.opt, params, normalize=False) # do not normalize eye_gaze.
        transform_scale_eye_gaze_video = transform_scale_nmfc_video
        transform_scale_rgb_video = get_transform(self.opt, params)
        if self.opt.use_shapes:
            # get_transform(self.opt, params, normalize=False) # do not normalize shape.
            transform_scale_shape_video = transform_scale_nmfc_video
        transform_scale_mask_video = get_transform(
            self.opt, params, normalize=False)

        # Read data.
        A_paths = []
        rgb_video = nmfc_video = shape_video = mask_video = eye_video = mouth_centers = eyes_centers = 0
        for i in range(self.opt.n_frames_G):
            j = min(local_idx - self.opt.n_frames_G + 1 + i, len(nmfc_video_paths)-1)
            # NMFC
            nmfc_video_path = nmfc_video_paths[j]
            nmfc_video_i = d.get_image(
                nmfc_video_path, transform_scale_nmfc_video)
            nmfc_video = nmfc_video_i if i == 0 else torch.cat(
                [nmfc_video, nmfc_video_i], dim=0)
            # RGB
            rgb_video_path = rgb_video_paths[j]
            rgb_video_i = d.get_image(
                rgb_video_path, transform_scale_rgb_video)
            rgb_video = rgb_video_i if i == 0 else torch.cat(
                [rgb_video, rgb_video_i], dim=0)
            # SHAPE
            if self.opt.use_shapes:
                shape_video_path = shape_video_paths[j]
                shape_video_i = d.get_image(
                    shape_video_path, transform_scale_shape_video)
                shape_video = shape_video_i if i == 0 else torch.cat(
                    [shape_video, shape_video_i], dim=0)
            # MASK
            mask_video_path = mask_video_paths[j]
            mask_video_i = d.get_image(
                mask_video_path, transform_scale_mask_video)
            mask_video = mask_video_i if i == 0 else torch.cat(
                [mask_video, mask_video_i], dim=0)
            A_paths.append(nmfc_video_path)
            if not self.opt.no_eye_gaze:
                landmark_video_path = landmark_video_paths[j]
                eye_video_i = create_eyes_image(landmark_video_path, first_nmfc_image.size,
                                                transform_scale_eye_gaze_video,
                                                add_noise=self.opt.isTrain)
                eye_video = eye_video_i if i == 0 else torch.cat(
                    [eye_video, eye_video_i], dim=0)
            if not self.opt.no_mouth_D and self.opt.isTrain:
                landmark_video_path = landmark_video_paths[j]
                mouth_centers_i = d.get_mouth_center(landmark_video_path)
                mouth_centers = mouth_centers_i if i == 0 else torch.cat(
                    [mouth_centers, mouth_centers_i], dim=0)
            if self.opt.use_eyes_D and self.opt.isTrain:
                landmark_video_path = landmark_video_paths[j]
                eyes_centers_i = d.get_eyes_center(landmark_video_path)
                eyes_centers = eyes_centers_i if i == 0 else torch.cat(
                    [eyes_centers, eyes_centers_i], dim=0)

        return_list = {'nmfc_video': nmfc_video, 'rgb_video': rgb_video, 'mask_video': mask_video, 'shape_video': shape_video,
                       'eye_video': eye_video, 'mouth_centers': mouth_centers, 'eyes_centers': eyes_centers, 'A_paths': A_paths}
        return return_list

    def __getitem__(self, index):
        index = np.random.randint(index*self.sample_freq, min(len(self.mdataset), (index+1)*self.sample_freq))
        i, param, label, img_info = self.mdataset[index]
        data = {'index': i, 'x_src': torch.FloatTensor(param), 'y_src': label}
        rdata = self._get_render_data(img_info)
        if rdata is not None:
            data.update(rdata)
        return data

    def get_pseudo(self, index, label, return_render_data=False):
        pseudo = self.mdataset.get_pseudo(index, label)
        if pseudo is None:
            return None
        param, dist, img_info, inter_param, inter_dist, inter_info = pseudo
        data = {'x_tgt': torch.FloatTensor(param), 'dist': dist, 'inter_tgt': torch.FloatTensor(inter_param), 'inter_dist': inter_dist, 'inter_img': self.get_img_path(inter_info)}
        if return_render_data:
            rdata = self._get_render_data(img_info)
            if rdata is not None:
                data.update(rdata)
        return data

class MultiDataset(data.Dataset):
    def __init__(self, opt, which='source', phase='train'):
        super().__init__()
        self.opt = opt
        self.which = which
        self.phase = phase
        self.actor_data = {}
        self.indices = []
        for actor in opt.selected_actors_wild:
            opt_ = deepcopy(opt)
            opt_.celeb = os.path.join(opt.celeb, actor)
            d = videoDataset()
            d.initialize(opt_)
            self.actor_data[actor] = d
            self.indices += [f'{actor} {i}' for i in range(len(d))]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        actor, idx = self.indices[index].split(' ')
        return self.actor_data[actor][int(idx)]

def get_train_loader(opt, which, prefix=None):
    dataset = SCMCLDataset(opt, which, prefix=prefix)
    return data.DataLoader(dataset=dataset,
                           batch_size=opt.batch_size,
                           shuffle=True,
                           num_workers=opt.nThreads,
                           drop_last=True,
                           pin_memory=True)


def get_wild_loader(opt, which):
    dataset = MultiDataset(opt, which)
    return data.DataLoader(dataset=dataset,
                           batch_size=opt.batch_size,
                           shuffle=True,
                           num_workers=opt.nThreads,
                           drop_last=True,
                           pin_memory=True)


def get_val_loader(opt, which):
    dataset = SCMCLDataset(opt, which, phase='val')
    return data.DataLoader(dataset=dataset,
                           batch_size=opt.batch_size,
                           shuffle=False,
                           num_workers=opt.nThreads,
                           drop_last=True,
                           pin_memory=True)

class InfiniteFetcher:
    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(self.loader)
    
    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            data = next(self.iter)
        Munch(**data)
        return data

class InputFetcher:
    def __init__(self, loader, loader_ref):
        self.loader = loader
        self.loader_ref = loader_ref

        self.iter = iter(self.loader)
        self.iter_ref = iter(self.loader_ref)

    def _fetch_inputs(self, label):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            data = next(self.iter)
        params = []
        inter_params = []
        masks = []
        dists = []
        inter_dists = []
        pseudos = []
        inds = data['index']
        for i, idx in enumerate(inds):
            pseudo = self.loader.dataset.get_pseudo(idx, int(label[i]), return_render_data=True)
            if pseudo is None:
                params.append(np.zeros_like(data['x_src'][i]))
                inter_params.append(np.zeros_like(data['x_src'][i]))
                masks.append(0)
                dists.append(np.inf)
                inter_dists.append(0)
            else:
                params.append(pseudo['x_tgt'])
                inter_params.append(pseudo['inter_tgt'])
                dists.append(pseudo['dist'])
                inter_dists.append(pseudo['inter_dist'])
                masks.append(1)
            pseudos.append(pseudo)
        params = torch.from_numpy(np.stack(params, axis=0))
        inter_params = torch.from_numpy(np.stack(inter_params, axis=0))
        masks = torch.from_numpy(np.array(masks, dtype=np.float32))
        dists = torch.from_numpy(np.array(dists, dtype=np.float32))
        inter_dists = torch.from_numpy(np.array(inter_dists, dtype=np.float32))
        data.update({'x_tgt': params, 'tgt_mask': masks, 'inter_tgt': inter_params, 
                     'pseudos': pseudos, 'dists': dists, 'inter_dists': inter_dists})
        return Munch(**data)

    def _fetch_refs(self):
        try:
            _, x, y = next(self.iter_ref)
        except StopIteration:
            self.iter_ref = iter(self.loader_ref)
            _, x, y = next(self.iter_ref)
        return x, y

    def __next__(self):
        x_ref, y_ref = self._fetch_refs()
        data = self._fetch_inputs(y_ref)
        data.update({'x_ref': x_ref, 'y_ref': y_ref})
        Munch(**data)
        return data


if __name__ == '__main__':
    from renderer.options.scmcl_options import TrainOptions
    opt = TrainOptions().parse()
    opt.train_root = 'MEAD'
    opt.celeb = 'celeb/train'
    opt.selected_actors = ['M003']
    opt.selected_emotions = ['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
    opt.dist_file = 'exp/similarity/dists/dists.pkl'
    loader = get_train_loader(opt, 'source')
    data = next(iter(loader))
    params = []
    masks = []
    pseudos = []
    inds = data['index']
    for i, idx in enumerate(inds):
        pseudo = loader.dataset.get_pseudo(idx, 1)
        if pseudo is None:
            param = np.zeros_like(data['x_src'][i])
            masks.append(0)
        else:
            param = pseudo['x_tgt']
            masks.append(1)
        params.append(param)
        pseudos.append(pseudo)
    params = torch.from_numpy(np.concatenate(params, axis=0))
    masks = torch.from_numpy(np.array(masks, dtype=np.float32))
    data.update({'x_tgt': params, 'tgt_mask': masks, 'pseudos': pseudos})
    print(data.keys())
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        elif isinstance(v, list):
            print(k, len(v))
        else:
            print(k, v)
    print(data['A_paths'])
