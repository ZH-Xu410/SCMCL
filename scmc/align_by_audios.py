import os
import cv2
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy import ndimage
import warnings

import librosa
import numpy as np
import python_speech_features


emotions = ['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
levels = [1, 3, 3, 3, 3, 3, 3]
emo_2_lvl = {e:l for e, l in zip(emotions, levels)}


def dtw(x, y, dist, warp=1, w=np.inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    assert np.isposinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not np.isposinf(w):
        D0 = np.full((r + 1, c + 1), np.inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = np.zeros((r + 1, c + 1))
        D0[0, 1:] = np.inf
        D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if (np.isposinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not np.isposinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = np.zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), np.zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


def _traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)


def align_audio(a1, a2, fps=30):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample_interval = 1 / fps
        sr = 16000
        x1, _ = librosa.load(a1, sr=sr)
        x2, _ = librosa.load(a2, sr=sr)
        mfcc1 = python_speech_features.mfcc(x1 , sr ,winstep=sample_interval)
        mfcc2 = python_speech_features.mfcc(x2 , sr ,winstep=sample_interval)
        dist, cost, acc_cost, path = dtw(mfcc1, mfcc2, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
        return path



def main():
    save_path = os.path.join(args.data_root, 'aligned_path.json')
    if os.path.exists(save_path):
        with open(save_path) as f:
            aligned_path = json.load(f)
    else:
        aligned_path = {}
    for actor in args.actors:
        aligned_path[actor] = {}
        paired_videos = {}
        for emo in args.emotions:
            if emo == 'disgusted' and os.path.exists(os.path.join(args.data_root, actor, 'video/front/', emo, f'level_{emo_2_lvl[emo]}', '030.mp4')):
                paired_videos['disgusted'] = ["028", "029", "030"]
            elif os.path.exists(os.path.join(args.data_root, actor, 'video/front/', emo, f'level_{emo_2_lvl[emo]}', '001.mp4')):
                paired_videos[emo] = ["001", "002", "003"]
            else:
                paired_videos[emo] = []
        
        for emo in args.emotions:
            if emo == 'neutral':
                continue
            if len(paired_videos[emo]) == 0:
                continue
                
            aligned_path[actor][emo] = {}
            for i in tqdm(range(3), desc=f'{actor} {emo}'):
                video_path = os.path.join(args.data_root, actor, 'video/front/', emo, f'level_{emo_2_lvl[emo]}', paired_videos[emo][i]+'.mp4')
                if not os.path.exists(video_path):
                    print(f'warning: file not exists {video_path}')
                else:
                    name1 = paired_videos['neutral'][i]
                    name2 = paired_videos[emo][i]
                    audio1 = os.path.join(args.data_root, actor, 'audio', 'neutral', 'level_1', name1+'.m4a')
                    audio2 = os.path.join(args.data_root, actor, 'audio', emo, f'level_{emo_2_lvl[emo]}', name2+'.m4a')
                    if not os.path.exists(audio1) or not os.path.exists(audio2):
                        continue
                    aligned_path[actor][emo][name1+'_'+name2] = list(map(lambda x: x.tolist(), align_audio(audio1, audio2)))
    
    with open(save_path, 'w') as f:
        json.dump(aligned_path, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',type=str,default='MEAD', help='data root')
    parser.add_argument('--actors', type=str, nargs='+', help='Subset of the MEAD actors', default=['M035', 'M023', 'W026', 'W025', 'W021', 'W019', 'M040', 'M011', 'W038', 'W023', 'W033', 'W040', 'M032', 'W036', 'M022', 'M039', 'W035', 'W016', 'M041', 'M027', 'M031', 'W014', 'M005', 'M019', 'M025', 'M042', 'M028', 'M037', 'M033', 'M024', 'W011', 'W028', 'W018', 'M034', 'M029', 'M007'])
    parser.add_argument('--emotions', type=str, nargs='+', help='Selection of Emotions', default=['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised'])
    args = parser.parse_args()

    main()
