
import torch
import torch.nn as nn
from loader.sampler import LabelSampler
from loader.data_loader import load_data_for_MultiDA
import time
import torch.nn.functional as F
import math
from utils.eval import test, predict
from utils import globalvar as gl
def train_for_UDA(args, model, optimizer, dataloaders):
    record_steps = 100
    pretrain = True
    DEVICE = gl.get_value('DEVICE')
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
    avg_cls_loss, avg_div_loss, avg_total_loss = 0, 0, 0
    early, best_acc, num_step = 0, 0, 0
    all_break = False
    for epoch in range(1, args.epochs+1):
        model.train()
        data_iter, num = [], []
        for data_loader in dataloaders['src_train']:
            data_iter.append(iter(data_loader))
            num.append(len(data_loader.dataset))
        max_num = max(num)
        data_tar_iter = iter(dataloaders['tar_train'])

        for step in range(0, max_num):
            if pretrain == True:
                if step > 0 and (step - 1) % len(dataloaders['tar_train']) == 0:
                    data_tar_iter = iter(dataloaders['tar_train'])
            src_data_list, src_label_list = [], []
            for src_data in data_iter:
                inputs, labels = next(src_data)
                src_data_list.append(inputs)
                src_label_list.append(labels)
            data_src = torch.cat(src_data_list)
            label_src = torch.cat(src_label_list)
            data_src, label_src = data_src.to(DEVICE), label_src.to(DEVICE)
            data_tar, _ = next(data_tar_iter)
            data_tar, _ = data_tar.to(DEVICE), _.to(DEVICE)
            
            domain_label_list = []
            for j in range(len(dataloaders['src_train'])):
                domain_label_list.append(torch.full((len(src_label_list[j]),1),j))
            domain_label_list.append(torch.full((len(_),1),j+1))
            domain_label = torch.cat(domain_label_list).to(DEVICE)

            outputs, div_loss = model(data_src, data_tar, label_src, domain_label)
            cls_loss = criterion(outputs, label_src)
            if num_step <= args.start_train:
                loss = cls_loss
            else:
                lambd = 2 / (1 + math.exp(-10 * (num_step - args.start_train) / args.lambd_step)) - 1
                loss = cls_loss + 0.5 * lambd * div_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                avg_div_loss += div_loss
                avg_cls_loss += cls_loss
                avg_total_loss += loss.item() 
            num_step += 1
            if num_step % record_steps == 0 : 
                print('step: [{:02d}/{:02d}] avg_div_loss: {:02f} avg_cls_loss: {:02f} loss: {:02f}'.format(num_step, args.steps, avg_div_loss/record_steps, avg_cls_loss/record_steps, avg_total_loss/record_steps))  
                avg_div_loss = 0
                avg_total_loss = 0
                avg_cls_loss = 0
            if num_step % args.save_steps == 0 : 
                print('step: [{:02d}/{:02d}]'.format(num_step, args.steps))  
                with torch.no_grad():
                    loss_tar, acc_tar = test(model, dataloaders['tar_test'])
                if best_acc < acc_tar :
                    best_acc = acc_tar
                    torch.save(model.state_dict(), './save/best(PT)_MIEM_{}to{}_{}.pth'.format(args.source, args.target, args.seed))
                    early = 0 
                early += 1
                if early > args.patience :
                    print('early stop')
                    all_break = True
                    break
            if num_step == args.start_train:
                pretrain = False
                select_class_num = args.batch_size // 2 
                while select_class_num > args.num_class:
                    select_class_num //= 2
                label_sampler = LabelSampler(args.update_steps, args.num_class, select_class_num)
                dataloaders['src_train'], dataloaders['tar_train'] = load_data_for_MultiDA(
                    args, args.root_dir, args.dataset, args.source, args.target, args.batch_size, label_sampler, pretrain=False)
            if num_step % args.update_steps == 0:
                pseudo_labels = predict(model, dataloaders['tar_test'])
                dataloaders['tar_train'].dataset.update_pseudo_labels(pseudo_labels)
                dataloaders['tar_train'].batch_sampler.label_sampler.set_batch_num(args.update_steps)
                for data_loader in dataloaders['src_train']:
                    data_loader.batch_sampler.label_sampler.set_batch_num(args.update_steps)
                data_iter, num = [], []
                for data_loader in dataloaders['src_train']:
                    data_iter.append(iter(data_loader))
                    num.append(len(data_loader.dataset))
                max_num = max(num)
                data_tar_iter = iter(dataloaders['tar_train'])
        if all_break:
            break
    print('final model save!')
    print('best_acc : {}'.format(best_acc))
    time_pass = time.time() - start_time
    print('Training {} epoch complete in {}h {}m {:.0f}s\n'.format(epoch, time_pass//3600, time_pass%3600//60, time_pass%60))
