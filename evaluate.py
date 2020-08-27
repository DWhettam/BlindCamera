import torch
import torchvision.models as model
import argparse
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from utils.datasets import AudioDataset
from utils.utils import *
import json
import itertools
import numpy as np
import pandas as pd
from collections import Counter

with open('runs/EPIC_baselines/run_16/args.txt') as json_file:
    model_args = json.load(json_file)

parser = argparse.ArgumentParser(description='BlindCamera_args')
parser.add_argument('--annotation_path', default='/mnt/storage/home/qc19291/scratch/EPIC/epic-kitchens-100-annotations/', type=str,
                    help='folder containing EPIC annotations')
parser.add_argument('--data_path', default='/mnt/storage/home/qc19291/scratch/EPIC/EPIC_audio.hdf5', type=str,
                    help='folder containing EPIC data')
parser.add_argument('--run_num', type=int, help='Which run to evaluate')
parser.add_argument('--gpus', default=2, type=int, help='number of gpus')

args = parser.parse_args()
print(args)

train_csv = args.annotation_path + 'EPIC_100_train.pkl'
evaluate_csv = args.annotation_path + 'EPIC_100_validation.pkl'
verb_classes = pd.read_csv(args.annotation_path + 'EPIC_100_verb_classes.csv')
noun_classes = pd.read_csv(args.annotation_path + 'EPIC_100_noun_classes.csv')
verb_classes = verb_classes.drop('instances', 1)
noun_classes = noun_classes.drop('instances', 1)

def create_csv(lbllist, predlist, label_list):
    lbllist = lbllist.numpy()
    predlist = predlist.numpy()
    
    c = Counter(lbllist)
    top20_classes = c.most_common(20)
    top20_classes = [i[0] for i in top20_classes]
    
    del_idx = [] 
    for idx, (lbl_item, pred_item) in enumerate(zip(lbllist, predlist)):
        if lbl_item not in top20_classes or pred_item not in top20_classes:
            del_idx.append(idx)

    for idx in sorted(del_idx, reverse=True):
        lbllist = np.delete(lbllist, idx)
        predlist = np.delete(predlist, idx)
        
    df = pd.DataFrame(lbllist, columns = ['labels'])
    df['predictions'] = predlist
    csv_path = 'runs/EPIC_baselines/run_' + str(args.run_num) + '/preds_labels.csv'
    df.to_csv('csv_path.csv', sep=',', encoding='utf-8')

def validate(net):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # Initialize the prediction and label lists(tensors)
    predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

    net.eval()
    end = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(VAL_LOADER):
        with torch.no_grad():
            # measure data loading time
            data_time.update(time.time() - end)
            
            inputs, targets = inputs.to(device), targets.to(device)

            inputs = inputs.contiguous().view(-1, 1, np.shape(inputs)[2], np.shape(inputs)[3])            
            
            outputs = model(inputs)
            
            outputs = outputs.reshape(int(model_args['batch_size']), 5, num_classes)
            outputs = torch.mean(outputs, 1)
            
            _, preds = torch.max(outputs, 1)

            #Append batch prediction results
            predlist=torch.cat([predlist,preds.view(-1).cpu()])
            lbllist=torch.cat([lbllist,targets.view(-1).cpu()])

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

    out = (' * Error@1 {top1.avg:.3f} Error@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    print(out)

    return losses.avg


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
                        batch_size=64, 
                        shuffle = False, 
                        drop_last = True, 
                        num_workers=28,
                        pin_memory=True)

model = models.resnet50()
with torch.no_grad():
    weights = torch.nn.Parameter(torch.mean(model._modules['conv1'].weight, 1, True))
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight = weights

num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(num_ftrs, 97))
model = torch.nn.DataParallel(model, device_ids = range(0, args.gpus))

model_path = 'runs/EPIC_baselines/run_' + str(args.run_num) + '/model.t7'
checkpoint = torch.load('model_path')
model.load_state_dict(checkpoint['state'])

validate(model)

if model_args['label_type'] == 'verb':    
    label_list = verb_classes.values.tolist() 
else:
    label_list = noun_classes.values.tolist()

image = create_csv(lbllist, predlist, label_list) 