import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from datetime import datetime

import argparse
import os
import wandb
import time
import json
from data.dataset import get_dataset
from data.preprocess import get_transform
from models.resnet_BFP import resnet_BFP

# from .trainer.trainer import Trainer, get_optimizer
# from .trainer.scheme import Scheme
from models.Q_modules import Q_Optimizer
from trainer import Trainer, get_optimizer, Scheduler, Q_Scheduler, Scheme, Q_Scheme, WandbLogger

parser = argparse.ArgumentParser()

### global arguments
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--trainer_config', type=str, default=None)
parser.add_argument('--wandb_project', type=str, default=None)

### logging arguments
parser.add_argument('--results_dir', default='./results', help='results dir')
parser.add_argument('--save', default='', help='saved folder')
parser.add_argument('--log_freq', type=int, default=10)

### dataset arguments
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--datapath', type=str, default='/home/wch/data/cifar100')

### model arguments
parser.add_argument('--model', default='resnet_BFP', choices=['resnet', 'resnet_BFP'])
parser.add_argument('--input_size', type=int, default=32)
parser.add_argument('--model_config', default='')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--workers', type=int, default=8)

### trainer arguments
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--optimizer', default='SGD')
parser.add_argument('--warm_up_epoch', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.1, help='init lr')

### BFPQ argument
parser.add_argument('--target_bit_W', type=int, default=2)
parser.add_argument('--target_bit_bA', type=int, default=2)
parser.add_argument('--K_update_mode', type=str, default='BinarySearch')

def main(args):
    print('Global Setting...')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    day_stamp = datetime.now().strftime('%Y-%m-%d')
    time_stamp = datetime.now().strftime('%H:%M:%S')

    if args.save == '':
        save_folder = args.model + '_' + time_stamp
    else:
        save_folder = args.save

    save_path = args.results_dir + '/' + day_stamp + '/' + save_folder

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    print('Save at ', save_path)
    with open(save_path + '/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    wandb_log = args.wandb_project is not None
    if wandb_log:
        if args.trainer_config is not None:
            wandb_config = {'config_save':save_path}
        else:
            wandb_config = args.__dict__

        print('Wandb Setting (Optional) ...')
        wandb.init(project=args.wandb_project, name=save_folder, config=wandb_config)

    print('-------- Data Loading ---------')
    train_transform = get_transform(args.dataset, 
                                    input_size=args.input_size, augment=True)
    test_transform = get_transform(args.dataset, 
                                   input_size=args.input_size, augment=False)
    
    train_set = get_dataset(args.dataset, split='train', 
                            transform=train_transform, 
                            datasets_path=args.datapath)
    test_set = get_dataset(args.dataset, split='val', 
                           transform=test_transform, 
                           datasets_path=args.datapath)
    
    train_loader = DataLoader(train_set, 
                              batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.workers, pin_memory=True)
    
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    
    print('--------- Model Creating ---------')

    model = resnet_BFP(depth=18, dataset='cifar100').to(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = get_optimizer(args.optimizer, model.parameters())
    scheme = Scheme(init_lr=args.lr, warm_up_epoch=args.warm_up_epoch)
    scheduler = Scheduler(optimizer=optimizer, scheme=scheme,
                          batches_per_epoch=len(train_loader))
    q_optimizer = Q_Optimizer(q_params_list=model.q_params_list())
    q_scheme = Q_Scheme(target_bit_bA=args.target_bit_bA, target_bit_W=args.target_bit_W,
                        K_update_mode=args.K_update_mode)
    q_scheduler = Q_Scheduler(q_optimizer=q_optimizer, q_scheme=q_scheme,
                              batches_per_epoch=len(train_loader))


    
    print('Trainer Creating...')
    if wandb_log:
        wandb_logger = WandbLogger()
    else:
        wandb_logger = None
    trainer = Trainer(model=model,
                    scheduler=scheduler, q_scheduler=q_scheduler,
                    criterion=criterion,
                    train_loader=train_loader, test_loader=test_loader,
                    device=args.device, log_freq=args.log_freq,
                    wandb_logger=wandb_logger)

    if args.trainer_config is not None:
        trainer.load_config(args.trainer_config)

    dummy_input = torch.zeros([args.batch_size, 3, args.input_size, args.input_size], device=args.device)
    trainer.register(dummy_input=dummy_input)

    trainer.save_config(save_dir=save_path)

    print('-------- Training --------')
    best_prec = 0
    for epoch in range(args.epochs):
        print('Train Epoch\t:', epoch)
        trainer.train(epoch)
        trainer.train_logger.log('END TRAIN')
        print('Test Epoch:\t', epoch)
        trainer.test(epoch)
        trainer.train_logger.log('END TEST')
        best_prec = max(best_prec, trainer.train_logger.top1.avg)
        if epoch % 5 == 0:
            model_dir = save_path + '/epoch_' + str(epoch)
            trainer.save_model(model_dir)

    print('--------- Training Done ---------')
    print(best_prec)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)