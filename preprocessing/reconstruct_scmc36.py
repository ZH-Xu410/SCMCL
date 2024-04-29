import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import torch
from skimage.transform import estimate_transform, warp
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DECA.decalib.datasets import detectors
from DECA.decalib.deca import DECA
from DECA.decalib.utils.config import cfg as deca_cfg

cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


def print_args(parser, args):
    message = ''
    message += '----------------- Arguments ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '-------------------------------------------'
    print(message)


def video2sequence(video_path):
    videofolder = video_path.rsplit('.', 1)[0]
    os.makedirs(videofolder, exist_ok=True)
    video_name = video_path.split('/')[-1].split('.')[0]
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        imagepath = '{}/{}_frame{:04d}.jpg'.format(videofolder, video_name, count)
        cv2.imwrite(imagepath, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list

class TestData:
    def __init__(self, video_path, iscrop=True, crop_size=224, scale=1.25, face_detector='fan', device='cuda'):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        self.capture = cv2.VideoCapture(video_path)
        # self.num_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # print('total {} images'.format(self.num_frames))
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        if face_detector == 'fan':
            self.face_detector = detectors.FAN(device=device)
        # elif face_detector == 'mtcnn':
        #     self.face_detector = detectors.MTCNN()
        else:
            print(f'please check the detector: {face_detector}')
            exit()

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center

    def __next__(self):
        ret, image = self.capture.read()
        if not ret:
            return None
        
        if len(image.shape) == 2:
            image = image[:,:,None].repeat(1,1,3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:,:,:3]

        h, w, _ = image.shape
        if self.iscrop:
            bbox, bbox_type = self.face_detector.run(image)
            if len(bbox) < 4:
                    print('no face detected! run original image')
                    left = 0; right = h-1; top=0; bottom=w-1
            else:
                    left = bbox[0]; right=bbox[2]
                    top = bbox[1]; bottom=bbox[3]
            old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
            size = int(old_size*self.scale)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        else:
            src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])

        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = image/255.

        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2,0,1)
        return torch.from_numpy(dst_image).float()
    
    def close(self):
        self.capture.release()


def main():
    print('---------- 3D face reconstruction (DECA) on MEAD database --------- \n')    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='Negative value to use CPU, or greater or equal than zero for GPU id.')
    parser.add_argument('--root', type=str, default='MEAD', help='Path to MEAD database.')
    parser.add_argument('--actors', type=str, nargs='+', help='Subset of the MEAD actors', default=['M035', 'M023', 'W026', 'W025', 'W021', 'W019', 'M040', 'M011', 'W038', 'W023', 'W033', 'W040', 'M032', 'W036', 'M022', 'M039', 'W035', 'W016', 'M041', 'M027', 'M031', 'W014', 'M005', 'M019', 'M025', 'M042', 'M028', 'M037', 'M033', 'M024', 'W011', 'W028', 'W018', 'M034', 'M029', 'M007'])
    parser.add_argument('--emotions', type=str, nargs='+', help='Subset of the MEAD actors', default=['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised'])
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()

    # Figure out the device
    gpu_id = int(args.gpu_id)
    if gpu_id < 0:
        device = 'cpu'
    elif torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            device = 'cuda:0'
        else:
            device = 'cuda:' + str(gpu_id)
    else:
        print('GPU device not available. Exit')
        exit(0)

    # Print Arguments
    print_args(parser, args)

    deca_cfg.model.use_tex = True
    deca = DECA(config = deca_cfg, device=device)

    emotions = ['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
    levels = [1, 3, 3, 3, 3, 3, 3]
    emo_2_lvl = {e:l for e, l in zip(emotions, levels)}

    if args.actors[0] == 'all':
        args.actors = [f for f in os.listdir(args.root) \
            if os.path.isdir(os.path.join(args.root, f))]
    if args.emotions[0] == 'all':
        args.emotions = emotions

    for j in range(args.rank, len(args.actors), args.world_size):
        actor = args.actors[j]
        data = {}
        save_path = os.path.join(args.root, actor + '_deca_scmcl.pkl')
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                data = pickle.load(f)
        
        # find videos to be processed; we only need frontal videos with maximum intensity of emotion
        actor_path = os.path.join(args.root, actor)
        if not os.path.exists(os.path.join(actor_path, 'video/front/angry/level_3')):
            continue
        # video_names = [f for f in os.listdir(os.path.join(actor_path, 'video/front/angry/level_3')) if f.endswith('.mp4')]
        for emotion in args.emotions:
            if emotion == 'disgusted':
                if os.path.exists(os.path.join(actor_path, 'video/front/disgusted/level_3/030.mp4')):
                    video_names = ["028.mp4", "029.mp4", "030.mp4"]
                elif os.path.exists(os.path.join(actor_path, 'video/front/disgusted/level_3/001.mp4')):
                    video_names = ["001.mp4", "002.mp4", "003.mp4"]
                else:
                    continue
            else:
                video_names = ["001.mp4", "002.mp4", "003.mp4"]
            data[emotion] = {}
            for v in tqdm(video_names, desc=f'{actor}-{emotion}'):
                video_path = os.path.join(actor_path, f'video/front/{emotion}/level_{emo_2_lvl[emotion]}', v)
                dataset = TestData(video_path, iscrop=True, face_detector='fan', scale=1.25, device=device)
                # run DECA
                params = []
                with torch.no_grad():
                    while True:
                        image = next(dataset)
                        if image is None:
                            break
                        image = image.to(device)[None,...]
                        codedict = deca.encode(image)
                        params.append(np.concatenate((codedict['pose'].cpu().numpy()[:,3:], codedict['exp'].cpu().numpy()), 1))   # jaw + expression params
                dataset.close()
                if len(params) == 0:
                    print(f'empty video {actor} {emotion} {v}')
                else:
                    data[emotion][v[:-4]] = np.concatenate(params, 0)
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f'{j+1}/{len(args.actors)}: {actor} processed')
    print('DONE!')


if __name__=='__main__':
    main()
