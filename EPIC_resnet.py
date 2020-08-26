from __future__ import print_function
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
import torch
import torchaudio
import torchvision
import time
import h5py
from audio_records import EpicAudioRecord
import matplotlib
from SpecAugment import spec_augment_pytorch
import datetime
import pickle
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PIL
import random
import os
from torchvision import transforms
from utils.plotcm import create_confusion_matrix
from utils.utils import *
from utils.datasets import AudioDataset
from torch.optim.lr_scheduler import *
import argparse

parser = argparse.ArgumentParser(description='BlindCamera_args')
parser.add_argument('--annotation_path', default='/mnt/storage/home/qc19291/scratch/EPIC/epic-kitchens-100-annotations/', type=str,
                    help='folder containing EPIC annotations')
parser.add_argument('--data_path', default='/mnt/storage/home/qc19291/scratch/EPIC/EPIC_audio.hdf5', type=str,
                    help='folder containing EPIC data')
parser.add_argument('--epochs', default = 32, type=int, help='number of epochs')
parser.add_argument('--batch_size', default = 32, type=int, help='Batch size')
parser.add_argument('--print_freq', default=10, type=int, help="print stats frequency")
parser.add_argument('--eval_freq', default=5, type=int, help="val evaluation frequency")
parser.add_argument('--ngpus', default=1, type=int, help='number of gpus')
parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning Rate')
parser.add_argument('--n_fft', default=2048, type=float, help='size of padded windowed signal in spectrogram')
parser.add_argument('--window_size', default=10, type=float, help='size of windowed signal in spectrogram without padding')
parser.add_argument('--hop_length', default=5, type=float, help='STFT hop length')
parser.add_argument('--sampling_rate', default=24000, type=float, help='audio sampling length')
parser.add_argument('--pretrained', default=True, type=bool, help='Imagenet pretraining')
parser.add_argument('--checkpoint', default=None, type=str, help='Model checkpointing')
parser.add_argument('--augment', default=True, type=bool, help='Audio data augmentations')
parser.add_argument('--scheduler', default=None, type=str, choices = ['MultiStep', 'Plateau'], help='Audio data augmentations')
parser.add_argument('--time_warp', default=20, type=int, help='Time warping parameter')
parser.add_argument('--freq_mask', default=30, type=int, help='Frequency masking parameter')
parser.add_argument('--time_mask', default=30, type=int, help='Time masking parameter')
parser.add_argument('--mask_size', default=10, type=int, help='Size of mask for frequency and time masking')
parser.add_argument('--mask_num', default=1, type=int, help='Number of time and frequency masks')
parser.add_argument('--clip_len', default = 1.279, type=float, help='Length of audio clips')
parser.add_argument('--num_mels', default=128, type=int, help='Number of mel frequency bins')
parser.add_argument('--Dropout', default=0.5, type=float, help='Dropout value')
parser.add_argument('--label_type', default='verb', type=str, choices=['verb', 'noun'], help='Label type: verb, noun')
parser.add_argument('--ops', default='SGD', type=str, choices=['SGD','Adam','AdamW'], help='Which optimiser to be used')

args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(map(str, list(range(0, args.ngpus))))
torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

train_csv = args.annotation_path + 'EPIC_100_train.pkl'
evaluate_csv = args.annotation_path + 'EPIC_100_validation.pkl'
verb_classes = pd.read_csv(args.annotation_path + 'EPIC_100_verb_classes.csv')
noun_classes = pd.read_csv(args.annotation_path + 'EPIC_100_noun_classes.csv')
verb_classes = verb_classes.drop('instances', 1)
noun_classes = noun_classes.drop('instances', 1)

if args.label_type == 'verb':
    num_classes = len(verb_classes.index)
else:
    num_classes = len(noun_classes.index)

previous_runs = os.listdir('runs/EPIC_baselines')
if len(previous_runs) == 0:
    run_number = 1
else:
    run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1

logdir = 'run_%02d' % run_number
writer = SummaryWriter(os.path.join('runs/EPIC_baselines', logdir))

with open(os.path.join(os.path.join('runs/EPIC_baselines', logdir), 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

def validate(net, epoch, checkpoint=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.eval()
    end = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(VAL_LOADER):
        with torch.no_grad():
            # measure data loading time
            data_time.update(time.time() - end)
            
            inputs, targets = inputs.to(device), targets.to(device)

            inputs = inputs.contiguous().view(-1, 1, np.shape(inputs)[2], np.shape(inputs)[3])            
            
            outputs = model(inputs)
            outputs = outputs.reshape(args.batch_size, 5, num_classes)
            outputs = torch.mean(outputs, 1)
            
            loss = criterion(outputs, targets)
            
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            err1 = 100. - prec1
            err5 = 100. - prec5

                
            losses.update(loss.item(), inputs.size(0))
            top1.update(err1.item(), inputs.size(0))
            top5.update(err5.item(), inputs.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            writer.add_scalar('loss/val', losses.val, epoch * len(TRAIN) + batch_idx)
            writer.add_scalar('error/val', top1.val, epoch * len(TRAIN) + batch_idx)
        
            if batch_idx % args.print_freq == 0:
                print('Validate:[{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    batch_idx, len(VAL_LOADER), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

        val_losses.append(losses.avg)
        val_errors.append(top1.avg)

    out = (' * Error@1 {top1.avg:.3f} Error@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    print(out)

    val_losses.append(losses.avg)
    val_errors.append(top1.avg)

    if checkpoint:
        print('Saving..')
        state = {
            'state': net.state_dict(),
            'epoch': epoch,
            'train_losses': train_losses,
            'train_errors': train_errors,
            'val_losses': val_losses,
            'val_errors': val_errors
        }
        print('SAVED!')
        checkpoint_path = os.path.join('runs/EPIC_baselines', logdir)
        checkpoint_path = os.path.join(checkpoint_path, 'model.t7')
        torch.save(state, checkpoint_path)

    return losses.avg



    
image_transform = torchvision.transforms.Compose([
    transforms.ToTensor(),
])

    
TRAIN = AudioDataset(args.data_path, 
                     train_csv, 
                     sampling_rate = args.sampling_rate, 
                     window_size = args.window_size, 
                     step_size = args.hop_length, 
                     n_fft = 512,
                     time_warp = args.time_warp,
                     freq_mask = args.freq_mask,
                     time_mask = args.time_mask,
                     mask_size = args.mask_size,
                     mask_num = args.mask_num,
                     augment = args.augment,
                     clip_len = args.clip_len,
                     label_type = args.label_type ,
                     mode = 'train',
                     im_transform=image_transform)

TRAIN_LOADER = DataLoader(dataset=TRAIN, 
                          batch_size=args.batch_size, 
                          shuffle=True, 
                          drop_last = True, 
                          num_workers=28,
                          pin_memory=True)


VAL = AudioDataset(args.data_path, 
                   evaluate_csv, 
                   sampling_rate = args.sampling_rate, 
                   window_size = args.window_size, 
                   step_size = args.hop_length, 
                   n_fft = 512,
                   time_warp = args.time_warp,
                   freq_mask = args.freq_mask,
                   time_mask = args.time_mask,
                   mask_size = args.mask_size,
                   mask_num = args.mask_num,
                   augment = args.augment,
                   clip_len = args.clip_len,
                   label_type = args.label_type,
                   mode = 'val',
                   im_transform=image_transform)

VAL_LOADER = DataLoader(dataset=VAL, 
                        batch_size=args.batch_size, 
                        shuffle = False, 
                        drop_last = True, 
                        num_workers=28,
                        pin_memory=True)

model = models.resnet50(pretrained=args.pretrained)
with torch.no_grad():
    weights = torch.nn.Parameter(torch.mean(model._modules['conv1'].weight, 1, True))
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight = weights

num_ftrs = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(args.Dropout),
    nn.Linear(num_ftrs, num_classes))

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model, device_ids = range(0, args.ngpus))

model.to(device)
model.train()
criterion = nn.CrossEntropyLoss()
if args.ops == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    if args.scheduler == 'MultiStep':
        scheduler = MultiStepLR(optimizer, milestones=[20,40], gamma=0.1)
    elif args.scheduler == 'Plateau':
        scheduler = ReduceLROnPlateau(optimizer, 'min')
elif args.ops == 'Adam':
    optimizer = optim.Adam(model.parameters())
elif args.ops == 'AdamW':
    optimizer = optim.AdamW(model.parameters())



val_losses = []
train_losses = []
val_errors = []
train_errors = []

class_accuracies = []

batch_time = AverageMeter()
data_time = AverageMeter()
losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()
end = time.time()
for epoch in range(args.epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(TRAIN_LOADER):
         # measure data loading time
        data_time.update(time.time() - end)
           
        inputs = inputs.to(device)
        outputs = model(inputs)

        targets = targets.to(device)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        err1 = 100. - prec1
        err5 = 100. - prec5

        losses.update(loss.item(), inputs.size(0))
        top1.update(err1[0], inputs.size(0))
        top5.update(err5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        writer.add_scalar('loss/train', losses.val, epoch * len(TRAIN) + batch_idx)
        writer.add_scalar('error/train', top1.val, epoch * len(TRAIN) + batch_idx)
                          
        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, batch_idx, len(TRAIN_LOADER), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

          
    train_losses.append(losses.avg)
    train_errors.append(top1.avg)
    if epoch % args.eval_freq == 0:
        val_loss = validate(model, epoch, args.checkpoint) 
        model.train()
    
    if args.scheduler == 'MultiStep':
        scheduler.step()
    elif args.scheduler == 'Plateau':
        scheduler.step(val_loss)
writer.close()

print('Finished Training')

