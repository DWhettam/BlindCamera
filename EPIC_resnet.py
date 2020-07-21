from __future__ import print_function
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
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
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PIL
import random
import os
from torchvision import transforms
from torch.nn.utils.rnn import pack_sequence
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
parser.add_argument('--n_fft', default=2048, type=float, help='size of padded windowed signal in spectrogram')
parser.add_argument('--window_size', default=10, type=float, help='size of windowed signal in spectrogram without padding')
parser.add_argument('--hop_length', default=5, type=float, help='STFT hop length')
parser.add_argument('--sampling_rate', default=24000, type=float, help='audio sampling length')
parser.add_argument('--pretrained', default=True, type=bool, help='Imagenet pretraining')
parser.add_argument('--augment', default=True, type=bool, help='Audio data augmentations')
parser.add_argument('--time_warp', default=20, type=int, help='Time warping parameter')
parser.add_argument('--freq_mask', default=30, type=int, help='Frequency masking parameter')
parser.add_argument('--time_mask', default=30, type=int, help='Time masking parameter')
parser.add_argument('--mask_size', default=10, type=int, help='Size of mask for frequency and time masking')
parser.add_argument('--num_mels', default=128, type=int, help='Number of mel frequency bins')
parser.add_argument('--label_type', default='full', type=str, help='Label type: full, verb, noun')

args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(map(str, list(range(0, args.ngpus))))
torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

train_csv = args.annotation_path + 'EPIC_100_train.pkl'
#test_csv = args.data_path + 'evaluation_setup/fold1_itest.csv'
evaluate_csv = args.annotation_path + 'EPIC_100_validation.pkl'

def validate(net, checkpoint=None):
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

            targets = targets.long()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if args.label_type != 'full':
                loss = criterion(outputs, targets)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                err1 = 100. - prec1
                err5 = 100. - prec5

            elif args.label_type == 'full':
                target = {k: v.to(device) for k, v in target.items()}
                loss_verb = criterion(outputs[0], target['verb'])
                loss_noun = criterion(outputs[1], target['noun'])
                loss = 0.5 * (loss_verb + loss_noun)
                verb_losses.update(loss_verb.item(), batch_size)
                noun_losses.update(loss_noun.item(), batch_size)

                verb_output = outputs[0]
                noun_output = outputs[1]

                verb_prec1, verb_prec5 = accuracy(verb_output, target['verb'], topk=(1, 5)) 
                verb_err1 = 100. - verb_prec1
                verb_err5 = 100. - verb_prec5
                verb_top1.update(verb_err1, batch_size)
                verb_top5.update(verb_err5, batch_size)

                noun_prec1, noun_prec5 = accuracy(noun_output, target['noun'], topk=(1, 5))
                noun_err1 = 100. - noun_prec1
                noun_err5 = 100. - noun_prec5
                noun_top1.update(noun_err1, batch_size)
                noun_top5.update(noun_err5, batch_size)

                prec1, prec5 = multitask_accuracy((verb_output, noun_output),
                                                  (target['verb'], target['noun']),
                                                  topk=(1, 5))
                err1 = 100. - prec1
                err5 = 100. - prec5



            losses.update(loss.item(), inputs.size(0))
            top1.update(err1[0], inputs.size(0))
            top5.update(err5[0], inputs.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.print_freq == 0:
                if args.label_type != 'full':
                    print('Validate:[{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        batch_idx, len(VAL_LOADER), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5))
                elif args.label_type == 'full':
                    print('Validate: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Verb Loss {verb_loss.val:.4f} ({verb_loss.avg:.4f})\t'
                          'Noun Loss {noun_loss.val:.4f} ({noun_loss.avg:.4f})\t'
                          'Verb Error@1 {verb_err1.val:.3f} ({verb_err1.avg:.3f})\t'
                          'Verb Error@5 {verb_err5.val:.3f} ({verb_err5.avg:.3f})\t'
                          'Noun Error@1 {noun_err1.val:.3f} ({noun_err1.avg:.3f})\t'
                          'Noun Error@5 {noun_err5.val:.3f} ({noun_err5.avg:.3f})\t'
                          'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        batch_idx, len(VAL_LOADER), batch_time=batch_time,
                        data_time=data_time, loss=losses, verb_loss = verb_losses, noun_loss = noun_losses,
                        verb_err1 = verb_top1, verb_err5 = verb_top5, noun_err1 = noun_top1,
                        noun_err5 = noun_top5, top1=top1, top5=top5))


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

def multitask_accuracy(outputs, labels, topk=(1,)):
    """
    Args:
        outputs: tuple(torch.FloatTensor), each tensor should be of shape
            [batch_size, class_count], class_count can vary on a per task basis, i.e.
            outputs[i].shape[1] can be different to outputs[j].shape[j].
        labels: tuple(torch.LongTensor), each tensor should be of shape [batch_size]
        topk: tuple(int), compute accuracy at top-k for the values of k specified
            in this parameter.
    Returns:
        tuple(float), same length at topk with the corresponding accuracy@k in.
    """
    max_k = int(np.max(topk))
    task_count = len(outputs)
    batch_size = labels[0].size(0)
    all_correct = torch.zeros(max_k, batch_size).type(torch.ByteTensor)
    if torch.cuda.is_available():
        all_correct = all_correct.cuda()
    for output, label in zip(outputs, labels):
        _, max_k_idx = output.topk(max_k, dim=1, largest=True, sorted=True)
        # Flip batch_size, class_count as .view doesn't work on non-contiguous
        max_k_idx = max_k_idx.t()
        correct_for_task = max_k_idx.eq(label.view(1, -1).expand_as(max_k_idx))
        all_correct.add_(correct_for_task)

    accuracies = []
    for k in topk:
        all_tasks_correct = torch.ge(all_correct[:k].float().sum(0), task_count)
        accuracy_at_k = float(all_tasks_correct.float().sum(0) * 100.0 / batch_size)
        accuracies.append(accuracy_at_k)
    return tuple(accuracies)


def my_collate(batch):
    # batch contains a list of tuples of structure (sequence, target)
    data = [item[0] for item in batch]
    data = pack_sequence(data, enforce_sorted=False)
    targets = [item[1] for item in batch]
    return [data, targets]

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
    
        spec = librosa.stft(audio_array, n_fft=511, hop_length=noverlap, win_length = nperseg)
        #spec = np.log(np.real(spec * np.conj(spec)) + eps)
        magnitude = np.abs(spec)**2
        spec = librosa.filters.mel(sr=24000, n_fft=511, n_mels=128)
        spec = spec.dot(magnitude)
        spec = librosa.power_to_db(spec, ref=np.max)
 
        return Image.fromarray(spec)

    def _trim(self, record):
        data_point = np.array(self.audio_files[record.untrimmed_video_name])
    
        start = record.start_timestamp
        end = record.stop_timestamp
        start = datetime.datetime.strptime(start, "%H:%M:%S.%f")
        end = datetime.datetime.strptime(end, "%H:%M:%S.%f")

        end_timedelta = end - datetime.datetime(1900, 1, 1)
        start_timedelta = start - datetime.datetime(1900, 1, 1)

        end_seconds = end_timedelta.total_seconds()
        start_seconds = start_timedelta.total_seconds()
    
        if (end_seconds - start_seconds) > 1.279:
            start = int(round(start_seconds*self.sampling_rate))
            end = int(round((end_seconds- 1.279)*self.sampling_rate))
            rdm_strt = random.randint(start, end)
            rdm_end = rdm_strt + int(np.floor((1.279*self.sampling_rate)))
            data_point = data_point[rdm_strt:rdm_end]
        else:
            mid_seconds = (start_seconds + end_seconds)/2
        
            left_seconds = mid_seconds - 0.639
            right_seconds = mid_seconds + 0.639
        
            left_sample = int(round(left_seconds * self.sampling_rate))
            right_sample = int(round(right_seconds * self.sampling_rate))
        
            duration = data_point.shape[0] / float(self.sampling_rate)
        
            if left_seconds < 0:
                data_point = data_point[:int(round(self.sampling_rate * 1.279))]
            elif right_seconds > duration:
                data_point = data_point[-int(round(self.sampling_rate * 1.279)):]
            else:
                data_point = data_point[left_sample:right_sample]

        return data_point

    def __getitem__(self, index):
        if self.audio_files is None:
            self.audio_files = h5py.File(self.hdf5_files, 'r')
        #get record
        record = self.audio_list[index]
        #trim to right action length
        data_point = self._trim(record)
          
        # getting spec
        spec = self._make_spec(data_point)
        
        # convert to tensor
        spec = self.im_transform(spec) 
        
        #augment
        if args.augment and self.mode == 'train':
            spec = spec_augment_pytorch.spec_augment(spec, 
                        time_warping_para = args.time_warp, frequency_masking_para = args.freq_mask, 
                        time_masking_para = args.time_mask)
       
        if self.label_type == 'full':
            return spec, record.label
 
        return spec, record.label[self.label_type]


    
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
                     n_fft = 511,
                     label_type = args.label_type ,
                     mode = 'train',
                     im_transform=image_transform)

TRAIN_LOADER = DataLoader(dataset=TRAIN, 
                          batch_size=args.batch_size, 
                          shuffle=True, 
                          drop_last = True, 
                          num_workers=8,
                          pin_memory=True)

VAL = AudioDataset(args.data_path, 
                   evaluate_csv, 
                   sampling_rate = args.sampling_rate, 
                   window_size = args.window_size, 
                   step_size = args.hop_length, 
                   n_fft = 511,
                   label_type = args.label_type,
                   mode = 'val',
                   im_transform=image_transform)

VAL_LOADER = DataLoader(dataset=VAL, 
                        batch_size=args.batch_size, 
                        shuffle = False, 
                        drop_last = True, 
                        num_workers=8,
                        pin_memory=True)

model = models.resnet50(pretrained=args.pretrained)
if args.pretrained:
    with torch.no_grad():
        weights = torch.nn.Parameter(torch.mean(model._modules['conv1'].weight, 1, True))
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1.weight = weights

num_ftrs = model.fc.in_features

if args.label_type == 'verb':
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 97)
    )
elif args.label_type == 'noun':
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 300)
    )

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model, device_ids = range(0, args.ngpus))

model.to(device)
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

val_losses = []
train_losses = []
val_errors = []
train_errors = []

batch_time = AverageMeter()
data_time = AverageMeter()
losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()
if args.label_type == 'full':
    verb_losses = AverageMeter()
    noun_losses = AverageMeter()
    verb_top1 = AverageMeter()
    verb_top5 = AverageMeter()
    noun_top1 = AverageMeter()
    noun_top5 = AverageMeter()

end = time.time()

for epoch in range(args.epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(TRAIN_LOADER):
         # measure data loading time
        data_time.update(time.time() - end)
           
        #targets = targets.long()
        inputs = inputs.to(device)
        outputs = model(inputs)

        if args.label_type != 'full':
            targets = targets.to(device)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            err1 = 100. - prec1
            err5 = 100. - prec5

        elif args.label_type == 'full':
            targets = {k: v.to(device) for k, v in targets.items()}
            loss_verb = criterion(outputs[0], targets['verb'])
            loss_noun = criterion(outputs[1], targets['noun'])
            loss = 0.5 * (loss_verb + loss_noun)
            verb_losses.update(loss_verb.item(), batch_size)
            noun_losses.update(loss_noun.item(), batch_size)

            verb_output = outputs[0]
            noun_output = outputs[1]

            verb_prec1, verb_prec5 = accuracy(verb_output, targets['verb'], topk=(1, 5)) 
            verb_err1 = 100. - verb_prec1
            verb_err5 = 100. - verb_prec5
            verb_top1.update(verb_err1, batch_size)
            verb_top5.update(verb_err5, batch_size)

            noun_prec1, noun_prec5 = accuracy(noun_output, targets['noun'], topk=(1, 5))
            noun_err1 = 100. - noun_prec1
            noun_err5 = 100. - noun_prec5
            noun_top1.update(noun_err1, batch_size)
            noun_top5.update(noun_err5, batch_size)

            prec1, prec5 = multitask_accuracy((verb_output, noun_output),
                                              (targets['verb'], targets['noun']),
                                              topk=(1, 5))
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

        if batch_idx % args.print_freq == 0:
            if args.label_type != 'full':
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, batch_idx, len(TRAIN_LOADER), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
            elif args.label_type == 'full':
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Verb Loss {verb_loss.val:.4f} ({verb_loss.avg:.4f})\t'
                      'Noun Loss {noun_loss.val:.4f} ({noun_loss.avg:.4f})\t'
                      'Verb Error@1 {verb_err1.val:.3f} ({verb_err1.avg:.3f})\t'
                      'Verb Error@5 {verb_err5.val:.3f} ({verb_err5.avg:.3f})\t'
                      'Noun Error@1 {noun_err1.val:.3f} ({noun_err1.avg:.3f})\t'
                      'Noun Error@5 {noun_err5.val:.3f} ({noun_err5.avg:.3f})\t'
                      'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, batch_idx, len(TRAIN_LOADER), batch_time=batch_time,
                    data_time=data_time, loss=losses, verb_loss = verb_losses, noun_loss = noun_losses,
                    verb_err1 = verb_top1, verb_err5 = verb_top5, noun_err1 = noun_top1,
                    noun_err5 = noun_top5, top1=top1, top5=top5))


    train_losses.append(losses.avg)
    train_errors.append(top1.avg)
    if epoch % args.eval_freq == 0:
        validate(model) 
        model.train()

print('Finished Training')

