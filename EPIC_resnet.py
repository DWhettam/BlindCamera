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
from utils.plotcm import plot_confusion_matrix
from torch.nn.utils.rnn import pack_sequence
from torch.optim.lr_scheduler import *
from sklearn.metrics import confusion_matrix
import argparse

parser = argparse.ArgumentParser(description='BlindCamera_args')
parser.add_argument('--annotation_path', default='/mnt/storage/home/qc19291/scratch/EPIC/epic-kitchens-100-annotations/', type=str,
                    help='folder containing EPIC annotations')
parser.add_argument('--data_path', default='/mnt/storage/home/qc19291/scratch/EPIC/EPIC_audio.hdf5', type=str,
                    help='folder containing EPIC data')
parser.add_argument('--epochs', default = 32, type=int, help='number of epochs')
parser.add_argument('--batch_size', default = 32, type=int, help='Batch size')
parser.add_argument('--print_freq', default=10, type=int, help="print stats frequency")
parser.add_argument('--eval_freq', default=1, type=int, help="val evaluation frequency")
parser.add_argument('--ngpus', default=1, type=int, help='number of gpus')
parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning Rate')
parser.add_argument('--n_fft', default=2048, type=float, help='size of padded windowed signal in spectrogram')
parser.add_argument('--window_size', default=10, type=float, help='size of windowed signal in spectrogram without padding')
parser.add_argument('--hop_length', default=5, type=float, help='STFT hop length')
parser.add_argument('--sampling_rate', default=24000, type=float, help='audio sampling length')
parser.add_argument('--pretrained', default=True, type=bool, help='Imagenet pretraining')
parser.add_argument('--augment', default=True, type=bool, help='Audio data augmentations')
parser.add_argument('--scheduler', default=None, type=str, choices = ['MultiStep', 'Plateau'], help='Audio data augmentations')
parser.add_argument('--time_warp', default=20, type=int, help='Time warping parameter')
parser.add_argument('--freq_mask', default=30, type=int, help='Frequency masking parameter')
parser.add_argument('--time_mask', default=30, type=int, help='Time masking parameter')
parser.add_argument('--mask_size', default=10, type=int, help='Size of mask for frequency and time masking')
parser.add_argument('--mask_num', default=1, type=int, help='Number of time and frequency masks')
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
#test_csv = args.data_path + 'evaluation_setup/fold1_itest.csv'
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
        torch.save(state, 'checkpoints/%s.t7' % checkpoint)

    return losses.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AudioDataset(Dataset):
    def __init__(self, audio_files, file_list, window_size, step_size, n_fft, label_type = 'full',  sampling_rate = 24000, mode = 'train', im_transform = None):
        if not os.path.exists(audio_files) or not os.path.exists(file_list):
            raise Exception('path does not exist')
        self.hdf5_files = audio_files
        self.audio_files = None
        self.file_list = file_list
        self.im_transform = im_transform
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.step_size = step_size
        self.n_fft = n_fft
        self.mode = mode
        self.n_mels = 128
        self.label_type = label_type
        
        self._parse_list()
        
    def _make_spec(self, audio_array):    
        eps = 1e-6

        nperseg = int(round(self.window_size * self.sampling_rate / 1e3))
        noverlap = int(round(self.step_size * self.sampling_rate / 1e3))
    
        spec = librosa.stft(audio_array, n_fft=512, hop_length=noverlap, win_length = nperseg)
        #spec = np.log(np.real(spec * np.conj(spec)) + eps)
        magnitude = np.abs(spec)**2
        spec = librosa.filters.mel(sr=24000, n_fft=512, n_mels=128)
        spec = spec.dot(magnitude)
        spec = librosa.power_to_db(spec, ref=np.max)
 
        return Image.fromarray(spec)

    def _trim(self, record):
        data_point = np.array(self.audio_files[record.untrimmed_video_name])
        data = [] 
        start = record.start_timestamp
        end = record.stop_timestamp
        start = datetime.datetime.strptime(start, "%H:%M:%S.%f")
        end = datetime.datetime.strptime(end, "%H:%M:%S.%f")

        end_timedelta = end - datetime.datetime(1900, 1, 1)
        start_timedelta = start - datetime.datetime(1900, 1, 1)

        end_seconds = end_timedelta.total_seconds()
        start_seconds = start_timedelta.total_seconds()
 
        if (end_seconds - start_seconds) > 1.279:
            if self.mode == 'val':                
                num_clips = 5
                clip_interval = int(round((((end_seconds - start_seconds) - 1.279) / num_clips) * self.sampling_rate))
                start = int(round(start_seconds * self.sampling_rate))
                for clip in range(num_clips):            
                    val_clip = data_point[start:(start+int(round((1.279 * self.sampling_rate))))]
                    data.append(val_clip)
                    start += clip_interval           
            else:
                start = int(round(start_seconds*self.sampling_rate))
                end = int(round((end_seconds- 1.279)*self.sampling_rate))
                rdm_strt = random.randint(start, end)
                rdm_end = rdm_strt + int(np.floor((1.279*self.sampling_rate)))
                data.append(data_point[rdm_strt:rdm_end])
        else:
            mid_seconds = (start_seconds + end_seconds)/2
        
            left_seconds = mid_seconds - 0.639
            right_seconds = mid_seconds + 0.639
        
            left_sample = int(round(left_seconds * self.sampling_rate))
            right_sample = int(round(right_seconds * self.sampling_rate))
        
            duration = data_point.shape[0] / float(self.sampling_rate)
        
            if left_seconds < 0:
                data.append(data_point[:int(round(self.sampling_rate * 1.279))])
            elif right_seconds > duration:
                data.append(data_point[-int(round(self.sampling_rate * 1.279)):])
            else:
                data.append(data_point[left_sample:right_sample])
        return data

    def __getitem__(self, index):
        if self.audio_files is None:
            self.audio_files = h5py.File(self.hdf5_files, 'r')
        #get record
        record = self.audio_list[index]
        #trim to right action length
        data = self._trim(record)
        
        specs = [] 
        for item in data: 
            # getting spec
            spec = self._make_spec(item)
        
            # convert to tensor
            spec = self.im_transform(spec) 
        
            #augment
            if args.augment and self.mode == 'train':
                spec = spec_augment_pytorch.spec_augment(spec, 
                            time_warping_para = args.time_warp, frequency_masking_para = args.freq_mask, 
                            time_masking_para = args.time_mask, frequency_mask_num = args.mask_num, time_mask_num = args.mask_num)
            
            spec = torch.squeeze(spec)
            specs.append(spec)
       
        specs = torch.stack(specs)
        
        if self.mode == 'val' and np.shape(specs)[0] == 1:
            specs = specs.repeat(5, 1, 1)  
           
        return specs, record.label[self.label_type]


    
    def __len__(self):
        return self.data_len
    
    def _parse_list(self):
        with open(self.file_list, 'rb') as f:
            data = pickle.load(f)
            self.audio_list = [EpicAudioRecord(tup) for tup in data.iterrows()]
            self.data_len = len(self.audio_list)


    
image_transform = torchvision.transforms.Compose([
    transforms.ToTensor(),
])

    
TRAIN = AudioDataset(args.data_path, 
                     train_csv, 
                     sampling_rate = args.sampling_rate, 
                     window_size = args.window_size, 
                     step_size = args.hop_length, 
                     n_fft = 512,
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
if args.pretrained:
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

conf_mats = []
class_accuracies = []

batch_time = AverageMeter()
data_time = AverageMeter()
losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()
end = time.time()


for epoch in range(args.epochs):  # loop over the dataset multiple times

    # Initialize the prediction and label lists(tensors)
    #predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    #lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

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

       


        ##-------Confusion Matrix Creation-------------

       # _, preds = torch.max(outputs, 1)

        # Append batch prediction results
        #predlist=torch.cat([predlist,preds.view(-1).cpu()])
        #lbllist=torch.cat([lbllist,targets.view(-1).cpu()])
       

    #if args.label_type == 'verb':    
     #   label_list = verb_classes.values.tolist()
    #else:
     #   label_list = noun_classes.values.tolist()

    ## Confusion matrix
    #conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
    #conf_mats.append(conf_mat)
    ## Per-class accuracy
    #class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    #class_accuracies.append(class_accuracy)
       
    #plot_buf = plot_confusion_matrix(conf_mat, label_list)
    #image = PIL.Image.open(plot_buf)
    #image = transforms.ToTensor()(image).unsqueeze(0)
    #grid = torchvision.utils.make_grid(image)
    #writer.add_image('Training confusion matrix', grid, 0)
#    writer.image("Training Confusion Matrix", image, step = epoch)
  
    train_losses.append(losses.avg)
    train_errors.append(top1.avg)
    if epoch % args.eval_freq == 0:
        val_loss = validate(model, epoch) 
        model.train()
    
    if args.scheduler == 'MultiStep':
        scheduler.step()
    elif args.scheduler == 'Plateau':
        scheduler.step(val_loss)
writer.close()
#print(conf_mats[-1])
#print(class_accuracies[-1])
print('Finished Training')

