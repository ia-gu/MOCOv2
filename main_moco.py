import math
import os
import random
import logging
import shutil
import mlflow
import hydra
from omegaconf import DictConfig

import moco.builder
import moco.loader
from moco.loader import get_dataloader
from moco.resnet import ResNet18

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.models as models
from tqdm import tqdm

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

@hydra.main(config_name='ssl_config', config_path='configs', version_base='1.1')
def main(cfg : DictConfig):
    mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(cfg.mlflow_runname)
    with mlflow.start_run():
        random_seed = cfg.train_parameters.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(hydra.utils.get_original_cwd()+'/weights/'+cfg.mlflow_runname+'/'+cfg.dataset+'/'+str(cfg.train_parameters.seed), exist_ok=True)
    gpu_ids=[]
    for i in range(torch.cuda.device_count()):
        gpu_ids.append(i)
    cfg.train_parameters.batch_size = int(cfg.train_parameters.batch_size)
    main_worker(device, gpu_ids, cfg)

def main_worker(device, gpu_ids, cfg):

    encoder = ResNet18 if cfg.dataset=='CIFAR10' else models.__dict__['resnet18']
    model = moco.builder.MoCo(
        encoder,
        cfg.moco.dim,
        cfg.moco.queue,
        cfg.moco.momentum,
        cfg.moco.temperature,
        cfg.moco.mlp
    )
    model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), cfg.train_parameters.lr, momentum=0.9, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    train_loader = get_dataloader(cfg)

    for epoch in range(cfg.train_parameters.n_epoch):
        adjust_learning_rate(optimizer, epoch, cfg)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, scaler)

        save_checkpoint(
            {"state_dict": model.module.state_dict()},
            is_best=False,
            filename=hydra.utils.get_original_cwd()+'/weights/'+cfg.mlflow_runname+'/'+cfg.dataset+'/'+str(cfg.train_parameters.seed)+'/checkpoint.pth'
        )

def train(train_loader, model, criterion, optimizer, epoch, device, scaler):
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    loop = tqdm(train_loader, unit='batch', desc='| Pretrain |', dynamic_ncols=True)
    for _, (images, _) in enumerate(loop):
        # measure data loading time
        images[0] = images[0].to(device, non_blocking=True)
        images[1] = images[1].to(device, non_blocking=True)
        optimizer.zero_grad()

        # compute output
        with torch.cuda.amp.autocast():
            output, target = model(im_q=images[0], im_k=images[1])
            loss = criterion(output, target)
        losses.update(loss.item(), images[0].size(0))
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    losses.update(loss.item(), images[0].size(0))
    progress.display(0)

def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth")

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate based on schedule"""
    lr = cfg.train_parameters.lr
    lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / cfg.train_parameters.n_epoch))
    # stepLR
    # lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print(f'current lr: {param_group["lr"]}')

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
if __name__ == "__main__":
    main()
