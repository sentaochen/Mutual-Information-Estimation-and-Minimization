
import torch
import numpy as np
import random
import os
import argparse

from loader.sampler import LabelSampler
from loader.data_loader import load_data_for_test, load_data_for_MultiDA
from model.model import MODEL
from utils.optimizer import get_optimizer
from utils.train_UDA import train_for_UDA
from utils import globalvar as gl
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='./data',
                    help='root dir of the dataset')     
parser.add_argument('--dataset', type=str, default='PACS',
                    help='the name of dataset')
parser.add_argument('--source', type=str, nargs='+', default=['art_painting', 'cartoon', 'photo'],
                    help='source domain')
parser.add_argument('--target', type=str, default='sketch',
                    help='target domain')
parser.add_argument('--net', type=str, default='resnet',
                    choices=['resnet18','resnet'],
                    help='which network to use')
parser.add_argument('--gpu', type=int, default=0,
                    help='gpu')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr_mult', type=float, nargs=4, default=[0.1, 0.1, 1, 1],
                    help='lr_mult (default: [0.1, 0.1, 1, 1])')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 200)')
parser.add_argument('--steps', type=int, default=50000,
                    help='number of steps to train (default: 50000)')
parser.add_argument('--save_steps', type=int, default=1000,
                    help='number of steps to record and save model (default: 1000)')
parser.add_argument('--update_steps', type=int, default=2000,
                    help='number of steps to update pseudo labels (default: 2000)')
parser.add_argument('--start_train', type=int, default=2000,
                    help='number of steps to start train (default: 2000)')
parser.add_argument('--lambd_step', type=int, default=20000,
                    help='number of steps to cal lambd (the training progress linearly changing from 0 to 1)')
parser.add_argument('--num_class', type=int, default=7,
                    help='number of class')
parser.add_argument('--save_check', type=bool, default=True,
                    help='save checkpoint or not(default: True)')
parser.add_argument('--patience', type=int, default=10,
                    help='early stopping to wait for improvment before terminating. (default: 10)')
parser.add_argument('--early', type=bool, default=True,
                    help='early stopping or not(default: True)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='MOMENTUM of SGD (default: 0.9)')
parser.add_argument('--decay', type=int, default=0.0005,
                    help='DECAY of SGD (default: 0.0005)') 
parser.add_argument("--batch_size", default=16, type=int, help="batch size") 
parser.add_argument('--seed', type=int, default=0,
                    help='seed')


# ===========================================================================================================================================================================
args = parser.parse_args()
# args.root_dir = '/data' # YOUR PATH



DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
gl._init()
gl.set_value('DEVICE', DEVICE)
bottleneck_dim = 1024

seed = args.seed
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    print(args)
    dataloaders = {}
    select_class_num = args.batch_size // 2 
    while select_class_num > args.num_class:
        select_class_num //= 2
    label_sampler = LabelSampler(args.start_train, args.num_class, select_class_num)
    
    dataloaders['src_train'], dataloaders['tar_train'] = load_data_for_MultiDA(
        args, args.root_dir, args.dataset, args.source, args.target, args.batch_size, label_sampler, pretrain=True)

    for ele in dataloaders['src_train']:
        print(len(ele.dataset))
    print(len(dataloaders['tar_train'].dataset))

    dataloaders['tar_test'] = load_data_for_test(
        args, args.root_dir, args.dataset, args.source, args.target, args.batch_size)

    print(len(dataloaders['tar_test'].dataset))
    
    model = MODEL(args.net, args.num_class, bottleneck_dim).to(DEVICE)
    optimizer = get_optimizer(model, args.lr, args.lr_mult)

    train_for_UDA(args, model, optimizer, dataloaders)
