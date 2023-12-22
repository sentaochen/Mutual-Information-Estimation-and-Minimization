import torch
import torch.nn as nn

from model.basenet import network_dict
from utils import globalvar as gl
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn
import torch.optim as optim
from torchmin import minimize
class MODEL(nn.Module):

    def __init__(self, basenet, n_class, bottleneck_dim):
        super(MODEL, self).__init__()
        self.basenet = network_dict[basenet]()
        self.basenet_type = basenet
        self._in_features = self.basenet.len_feature()
        
        if self.basenet_type.lower() not in ['resnet18']:
            self.bottleneck = nn.Sequential(
                nn.Linear(self._in_features, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            )
            self.bottleneck[0].weight.data.normal_(0, 0.005)
            self.bottleneck[0].bias.data.fill_(0.1)
            self.fc = nn.Sequential(
                sn(nn.Linear(bottleneck_dim, bottleneck_dim)),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                sn(nn.Linear(bottleneck_dim, n_class))
            )
            self.fc[0].weight.data.normal_(0, 0.01)
            self.fc[0].bias.data.fill_(0.0)

            self.fc[-1].weight.data.normal_(0, 0.01)
            self.fc[-1].bias.data.fill_(0.0)
        else:
            print('use the shallower net, type:', basenet)
            self.fc = nn.Linear(self._in_features, n_class)


    def forward(self, source, target, label_src, domain_label):
        DEVICE = gl.get_value('DEVICE')
        source_features = self.basenet(source)
        if self.basenet_type.lower() not in ['resnet18']:
            source_features = self.bottleneck(source_features)
        target_features = self.basenet(target)
        if self.basenet_type.lower() not in ['resnet18']:
            target_features = self.bottleneck(target_features)
        softmax_layer = nn.Softmax(dim=1).to(DEVICE)
        target_softmax = softmax_layer(self.fc(target_features))
        target_prob, target_pseudolabel = torch.max(target_softmax, 1)
        label = torch.cat((label_src, target_pseudolabel), dim=0)
        loss = MI(source_features, target_features, label, domain_label)
        return self.fc(source_features), loss

    
    def get_bottleneck_features(self, inputs):
        features = self.basenet(inputs)
        return self.bottleneck(features)

    def get_fc_features(self, inputs):
        features = self.basenet(inputs)
        if self.basenet_type.lower() not in ['resnet18']:
            features = self.bottleneck(features)
        return self.fc(features)



def pairwise_distances(x, y, power=2, sum_dim=2):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n,m,d)
    y = y.unsqueeze(0).expand(n,m,d)
    dist = torch.pow(x-y, power).sum(sum_dim)
    return dist

def StandardScaler(x,with_std=False):
    mean = x.mean(0, keepdim=True)
    std = x.std(0, unbiased=False, keepdim=True)
    x -= mean
    if with_std:
        x /= (std + 1e-10)
    return x

def MI(source_features, target_features, y,l,sigma=None,lamda=1e-2): 
    DEVICE = gl.get_value('DEVICE')
    if sigma is None:
        pairwise_dist = torch.cdist(source_features,source_features,p=2)**2 
        sigma = torch.median(pairwise_dist[pairwise_dist!=0])
    X = torch.cat((source_features,target_features),dim=0)
    X_norm = torch.sum(X ** 2, axis=-1).to(DEVICE)
    l = torch.squeeze(l,-1)
    Deltay = (y[:,None]==y) * 1.0 
    Deltal = (l[:,None]==l) * 1.0
    K_W = torch.exp(-( X_norm[:, None] + X_norm[None,:] - 2 * torch.mm(X, X.T)) / sigma) * Deltay
    def Obj(theta):
        div = torch.mean(torch.matmul(K_W * Deltal,theta)) - torch.mean(torch.exp(torch.matmul(theta * K_W,Deltal) - 1))
        reg = lamda * torch.matmul(theta,theta) 
        return - div + reg

    theta_0 = torch.zeros(X.shape[0], device=DEVICE)
    result = minimize(Obj,theta_0,method='l-bfgs')
    theta_hat = result.x
    div = torch.mean(torch.matmul(K_W * Deltal,theta_hat)) - torch.mean(torch.exp(torch.matmul(theta_hat * K_W,Deltal) - 1))
    return div 
       
