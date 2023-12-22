import os
import torch
from torchvision import datasets, transforms

from loader.data_list import ImageList
from loader.sampler import BatchSampler

def load_data_for_test(args, root_dir, dataset, src_domain, tar_domain, batch_size, label_sampler=None):
    crop_size = 224
    resize_size = 256
    transform_dict = {
        'train': transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])}

    list_root_dir = os.path.join(root_dir,dataset, 'list')
    data_root_dir = os.path.join(root_dir, dataset)
    # data_list file name
    unlabeled_target_list = os.path.join(list_root_dir, '{}.txt'.format(tar_domain))
    
    tar_data = ImageList(unlabeled_target_list, data_root_dir, transform=transform_dict['train'])
    
    tar_loader = torch.utils.data.DataLoader(tar_data, batch_size=batch_size, shuffle=False, drop_last=False,
                                            num_workers=4)
    return tar_loader
    

def load_data_for_MultiDA(args, root_dir, dataset, src_domains , tar_domain, batch_size, label_sampler=None, pretrain=None):
    crop_size = 224
    resize_size = 256
    transform_dict = {
        'train': transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])}

    list_root_dir = os.path.join(root_dir, dataset, 'list')
    data_root_dir = os.path.join(root_dir, dataset)
    unlabeled_target_list = os.path.join(list_root_dir, '{}.txt'.format(tar_domain))

    labeled_source_list_array = []
    for src in src_domains:
        if src :
            labeled_source_list_array.append(os.path.join(list_root_dir, '{}.txt'.format(src)))

    src_loader_labeled_list = []
    domain_label = 0
    for labeled_source_list in labeled_source_list_array:
        src_data_labeled = ImageList(labeled_source_list, data_root_dir, transform=transform_dict['train'], type='labeled')
        src_batch_sampler_labeled = BatchSampler(src_data_labeled, label_sampler, batch_size )
        src_loader_labeled_list.append(torch.utils.data.DataLoader(src_data_labeled,batch_sampler = src_batch_sampler_labeled, num_workers=4))
        domain_label += 1

    if pretrain==True:
        tar_data_unlabeled = ImageList(unlabeled_target_list, data_root_dir, transform=transform_dict['test'], type='labeled')
        tar_loader_unlabeled = torch.utils.data.DataLoader(tar_data_unlabeled, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    else:
        tar_data_unlabeled = ImageList(unlabeled_target_list, data_root_dir, transform=transform_dict['test'], type='unlabeled')
        tar_batch_sampler_unlabeled = BatchSampler(tar_data_unlabeled, label_sampler, batch_size)
        tar_loader_unlabeled = torch.utils.data.DataLoader(tar_data_unlabeled, batch_sampler = tar_batch_sampler_unlabeled, num_workers=4)

    return src_loader_labeled_list, tar_loader_unlabeled