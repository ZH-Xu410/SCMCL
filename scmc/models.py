import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet


def distance(out, average=True):
    # same class
    dist0 = []
    for i in range(out.shape[1]):
        for j in range(out.shape[1]):
            if j == i:
                continue
            dist0.append(torch.norm(out[:, i]-out[:, j], p=2, dim=1))
    dist0 = torch.cat(dist0, dim=0)

    # different class
    dist1 = []
    for i in range(out.shape[0]):
        for j in range(out.shape[0]):
            if j == i:
                continue
            dist1.append(torch.norm(out[i]-out[j], p=2, dim=1))
    dist1 = torch.cat(dist1, dim=0)

    if average:
        dist0 = dist0.mean()
        dist1 = dist1.mean()

    return dist0, dist1


def similarity(out, average=True):
    # same class
    sim0 = []
    for i in range(out.shape[1]):
        for j in range(out.shape[1]):
            if j == i:
                continue
            sim0.append(F.cosine_similarity(out[:, i], out[:, j], dim=1))
    sim0 = torch.cat(sim0, dim=0)

    # different class
    sim1 = []
    for i in range(out.shape[0]):
        for j in range(out.shape[0]):
            if j == i:
                continue
            sim1.append(F.cosine_similarity(out[i], out[j], dim=1))
    sim1 = torch.cat(sim1, dim=0)

    if average:
        sim0 = sim0.mean()
        sim1 = sim1.mean()

    return sim0, sim1

class MDist:
    def __init__(self, mode='cosine', momentum=0.999, warmup=500):
        self.mode = mode
        self.momentum = momentum
        self.warmup = warmup

        self.pos_dist = 0.
        self.neg_dist = 0.
        self.step = 0
    
    @torch.no_grad()
    def update(self, out, scale=None):
        if self.mode == 'cosine':
            sim0, sim1 = similarity(out)
            dist = (1 - sim0, 1 - sim1)
        elif self.mode == 'l2':
            dist = distance(out)
        else:
            raise ValueError("Invalid mode.")
        
        if scale is None:
            scale = lambda x: x
        if self.step < self.warmup:
            momentum = np.linspace(0, self.momentum, self.warmup)[self.step]
            self.step += 1
        else:
            momentum = self.momentum
        self.pos_dist = self.pos_dist * momentum + (1 - momentum) * scale(dist[0]).item()
        self.neg_dist = self.neg_dist * momentum + (1 - momentum) * scale(dist[1]).item()


class ImageEncoder(nn.Module):
    def __init__(self, depth=18, dim=128, fc_dim=None):
        super().__init__()
        self.model = getattr(resnet, f'resnet{depth}')(pretrained=True)
        if fc_dim is None:
            fc_dim = self.model.fc.weight.shape[1]
        self.model.fc = nn.Linear(fc_dim, dim)
        self.dim = dim
        self.act = nn.Tanh()

    def forward(self, x):
        return self.proj(self.encode(x))

    def encode(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        return x
    
    def proj(self, x):
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        if x.shape[1] != self.dim:
            x = self.model.fc(x)
        x = self.act(x)
        
        return x
    
    @property
    def feat_dim(self):
        return self.model.fc.in_features

class AudioEncoder(nn.Module):
    def __init__(self, nconv=14, dim=128, nchan=64, chin=1, downsample_freq=2, fc_dim=None):
        super().__init__()
        self.nconv = nconv
        convs = []
        for iconv in range(nconv):
            if (iconv+1) % 4 == 0:
                nchan *= 2
            if iconv % downsample_freq == 0:
                stride = 2
            else:
                stride = 1
            convs.append(nn.Sequential(*[
                nn.Conv1d(chin,nchan,3,stride=stride,padding=1,bias=False),
                nn.BatchNorm1d(nchan),
                nn.ReLU(inplace=True)
            ]))
            chin = nchan
        self.convs = nn.Sequential(*convs)
        if fc_dim is None:
            fc_dim = nchan
        self.fc = nn.Linear(fc_dim, dim)
        self.dim = dim
        self.act = nn.Tanh()

    def forward(self, x):
        return self.proj(self.encode(x))
    
    def encode(self, x):
        return self.convs(x)
    
    def proj(self, x):
        x = x.mean(-1)
        if x.shape[1] != self.dim:
            x = self.fc(x)
        x = self.act(x)
        return x
    
    @property
    def feat_dim(self):
        return self.fc.in_features


class ExprEncoder(AudioEncoder):
    def __init__(self, nconv=14, dim=128, nchan=64, chin=1, downsample_freq=4, fc_dim=None):
        super().__init__(nconv, dim, nchan, chin, downsample_freq, fc_dim=fc_dim)


class Loss(nn.Module):
    def __init__(self, mode='l2', device=torch.device("cpu"), nframe=1):
        super(Loss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mode = mode
        self.device = device
        self.nframe = nframe

    def forward(self, x):
        """
        x: (k/j (classes), m/i (samples), c (n_features))
        """

        x1 = x[:, 0, :].reshape(-1, x.shape[-1]).contiguous()
        # x2 = x[:, 1:, :].reshape(-1, x.shape[-1]).contiguous()
        
        if self.mode == "l2":
            loss = 0.
            for i in range(self.nframe):
                x2 = x[:, i+1, :].reshape(-1, x.shape[-1]).contiguous()
                dist = torch.cdist(x1, x2, p=2)
                sims = -dist
                labels = torch.tensor(np.arange(x.size(0))).to(self.device)
                loss += self.ce(sims, labels)
                if i == 0:
                   dist_out = dist[:, labels] 
        
        elif self.mode == "cosine":
            temperature = 0.1
            loss = 0.
            for i in range(self.nframe):
                x2 = x[:, i+1, :].reshape(-1, x.shape[-1]).contiguous()
                x1 = F.normalize(x1, dim=1)
                x2 = F.normalize(x2, dim=1)
                sims = x1 @ x2.T
                dist = 1 - sims
                labels = torch.tensor(np.arange(x.size(0))).to(self.device)
                loss += self.ce(sims/temperature, labels)
                if i == 0:
                   dist_out = dist[:, labels]

        else:
            raise ValueError("Invalid mode.")

        return loss, dist_out


class ScaleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)
    
    def forward(self, x):
        return self.w * x + self.b


class SyncNet(nn.Module):
    def __init__(self, num_layers_in_fc_layers = 1024):
        super(SyncNet, self).__init__()

        self.__nFeatures__ = 24
        self.__nChs__ = 32
        self.__midChs__ = 32

        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=(1,1)),

            nn.Conv2d(64, 192, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,2)),

            nn.Conv2d(192, 384, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            
            nn.Conv2d(256, 512, kernel_size=(5,4), padding=(0,0)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.netfcaud = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )

        self.netfclip = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )

        self.netcnnlip = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5,7,7), stride=(1,2,2), padding=0),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),

            nn.Conv3d(96, 256, kernel_size=(1,5,5), stride=(1,2,2), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),

            nn.Conv3d(256, 512, kernel_size=(1,6,6), padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )

    def forward_aud(self, x):

        mid = self.netcnnaud(x) # N x ch x 24 x M
        mid = mid.view((mid.size()[0], -1)) # N x (ch x 24)
        out = self.netfcaud(mid)

        return out

    def forward_lip(self, x):

        mid = self.netcnnlip(x) 
        mid = mid.view((mid.size()[0], -1)) # N x (ch x 24)
        out = self.netfclip(mid)

        return out

    def forward_lipfeat(self, x):

        mid = self.netcnnlip(x)
        out = mid.view((mid.size()[0], -1)) # N x (ch x 24)

        return out