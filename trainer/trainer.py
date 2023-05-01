import torch
import torch.nn as nn
import torch.optim as optim
from utils import meters
from .scheduler import Scheduler
from .Q_scheduler import Q_Scheduler
import time
import json

def get_optimizer(optimizer_name, params):
    if optimizer_name == 'SGD':
        return optim.SGD(params=params, lr=0.1, momentum=0.9, weight_decay=5e-4)
    elif optimizer_name == 'Adam':
        return optim.Adam(params=params)

class TrainingLog:
    '''
    Training Log
    '''
    def __init__(self) -> None:
        self.batch_time = meters.AverageMeter()
        self.data_time = meters.AverageMeter()
        self.losses = meters.AverageMeter()
        self.top1 = meters.AverageMeter()
        self.top5 = meters.AverageMeter()

    def reset(self):
        self.batch_time.reset()
        self.data_time.reset()
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()

    def log(self, batch, logfile=None):
        print("Batch:{} Time:{:.3f}({:.3f})\tDataTime:{:.3f}({:.3f})\tloss:{:.4f}({:.4f})\tprec@1:{:.4f}({:.4f})\tprec@5:{:.4f}({:.4f})".format(batch, 
                                  self.batch_time.val, self.batch_time.avg,
                                  self.data_time.val, self.data_time.avg,
                                  self.losses.val, self.losses.avg, 
                                  self.top1.val, self.top1.avg,
                                  self.top5.val, self.top5.avg))
    
class Trainer:
    '''
    Trainer:
    '''
    def __init__(self, model, scheduler:Scheduler, q_scheduler:Q_Scheduler, criterion, 
                 train_loader, test_loader, device='cuda:0', 
                 training_log:TrainingLog=TrainingLog(), log_freq=10):
        ### 
        self.model = model
        self.scheduler = scheduler
        self.q_scheduler = q_scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.training_log = training_log
        self.log_freq = log_freq
        self.best_prec = 0
        ### 

    def register(self, dummy_input):
        self.model.register()
        dummy_output = self.model(dummy_input)
        self.q_scheduler.register()
        print('Q_Scheduler Register Done.')

    def train(self, epoch):
        self.forward(epoch=epoch, dataloader=self.train_loader, train=True)

    def test(self, epoch):
        self.forward(epoch=epoch, dataloader=self.test_loader, train=False)

    def forward(self, epoch, dataloader, train=True):
        
        if train:
            self.model.train()
            self.q_scheduler.zero_sensitivity()
        else:
            self.model.eval()

        self.training_log.reset()
        
        time_stamp = time.time()
        for batch, (inputs, labels) in enumerate(dataloader):
            
            self.training_log.data_time.update(time.time()-time_stamp)
            time_stamp = time.time()

            self.scheduler.zero_grad()
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            
            if train:
                loss.backward()
                self.scheduler.step()

            prec = meters.accuracy(outputs.detach(), labels, (1, 5))
            
            self.training_log.batch_time.update(time.time()-time_stamp)
            time_stamp = time.time()

            self.training_log.losses.update(loss.detach().cpu())
            self.training_log.top1.update(prec[0])
            self.training_log.top5.update(prec[1])
            
            if batch % self.log_freq == 0:
                self.training_log.log(batch)

        if train:
            self.scheduler.update_epoch()
            self.q_scheduler.step()

        print('Epoch: {}, lr: {}'.format(epoch, self.scheduler.lr))

    def save_config(self, save_dir):
        trainer_config_path = save_dir + '/trainer_config.json'
        with open(trainer_config_path, 'w') as f:
            train_config = {
                'Common':self.scheduler.scheme.__dict__,
                'Quantization':self.q_scheduler.q_scheme.__dict__,
            }
            json.dump(train_config, f, indent=4)
            print('Successful Dumping Trainer Config to '+ trainer_config_path)

    def load_config(self, trainer_config_path):
        with open(trainer_config_path, 'r') as f:
            trainer_config = json.load(f)
            print('Successful Loading Trainer Config from ' + trainer_config_path)

    def save_state(self, save_dir):
        trainer_state = save_dir + '/trainer_state.npy' 

    def load_state(self, trainer_state_path):
        trainer_state = trainer_state_path

    def save_model(self, save_dir):
        pass

    def load_model(self, model_path):
        pass