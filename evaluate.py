import torch
import torchvision.models as model
import argparse
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from utils.datasets import AudioDataset
from utils.utils import *
import torchvision
from torchvision import transforms
from torchvision import models
import json
import itertools
import numpy as np
import pandas as pd
import time
from collections import Counter
import os
import pickle
import copy

parser = argparse.ArgumentParser(description='BlindCamera_args')
parser.add_argument('--annotation_path', default='/mnt/storage/home/qc19291/scratch/EPIC/epic-kitchens-100-annotations/', type=str,
                    help='folder containing EPIC annotations')
parser.add_argument('--data_path', default='/mnt/storage/home/qc19291/scratch/EPIC/EPIC_audio.hdf5', type=str,
                    help='folder containing EPIC data')
parser.add_argument('--noun_run', type=str, default = None,  help='noun run to evaluate')
parser.add_argument('--verb_run', type=str, default = None, help='verb run to evaluate')
parser.add_argument('--ngpus', default=1, type=int, help='number of gpus')

args = parser.parse_args()
print(args)

if args.verb_run is not None:
    with open('runs/EPIC_baselines/run_'+args.verb_run+'/args.txt') as json_file:
        verb_args = json.load(json_file)
if args.noun_run is not None:
    with open('runs/EPIC_baselines/run_'+args.noun_run+'/args.txt') as json_file:
        noun_args = json.load(json_file)

os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(map(str, list(range(0, args.ngpus))))
torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_csv = args.annotation_path + 'EPIC_100_train.pkl'
evaluate_csv = args.annotation_path + 'EPIC_100_validation.pkl'
verb_classes = pd.read_csv(args.annotation_path + 'EPIC_100_verb_classes.csv')
noun_classes = pd.read_csv(args.annotation_path + 'EPIC_100_noun_classes.csv')
verb_classes = verb_classes.drop('instances', 1)
noun_classes = noun_classes.drop('instances', 1)


num_verb_classes = len(verb_classes.index)
num_noun_classes = len(noun_classes.index)

def create_csv(verb_lbllist = None, verb_predlist=None, verb_outlist=None, verb_run=None, noun_lbllist = None, noun_predlist = None, noun_outlist=None, noun_run = None):
    
    if verb_run is not None:
        new_dim = int(np.shape(verb_outlist)[0]) * int(np.shape(verb_outlist)[1])
        verb_outlist = verb_outlist.view(new_dim, np.shape(verb_outlist)[2]).cpu()
        verb_lbllist = verb_lbllist.cpu()
        verb_predlist = verb_predlist.cpu()
        verb_predlist = verb_predlist.numpy()


        df_preds = pd.DataFrame(verb_predlist, columns = ['predictions'])
        df_labels = pd.DataFrame(verb_lbllist, columns = ['labels'])
        csv_path = 'runs/EPIC_baselines/run_' + str(verb_run)
        df_preds.to_csv(csv_path + '/preds.csv', sep=',', encoding='utf-8')
        df_labels.to_csv(csv_path + '/labels.csv', sep=',', encoding='utf-8')


    if noun_run is not None:
        new_dim = int(np.shape(noun_outlist)[0]) * int(np.shape(noun_outlist)[1])
        noun_outlist = noun_outlist.view(new_dim, np.shape(noun_outlist)[2]).cpu()
        noun_lbllist = noun_lbllist.cpu()
        noun_predlist = noun_predlist.cpu()
        noun_predlist = noun_predlist.numpy()
   
        df_labels = pd.DataFrame(noun_lbllist, columns = ['labels'])
        df_preds = pd.DataFrame(noun_predlist, columns = ['predictions'])
        csv_path = 'runs/EPIC_baselines/run_' + str(noun_run)
        df_preds.to_csv(csv_path + '/preds.csv', sep=',', encoding='utf-8')
        df_labels.to_csv(csv_path + '/labels.csv', sep=',', encoding='utf-8')

    if verb_run is not None and noun_run is not None:
        verb_outlist = verb_outlist.numpy()
        noun_outlist = noun_outlist.numpy()
        scores = {'scores':{'verb':verb_outlist,'noun':noun_outlist}}   
        with open('scores.pkl', 'wb') as handle:
            pickle.dump(scores, handle) 

def validate(net, num_classes, VAL_LOADER, model_args):
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # Initialize the prediction and label lists(tensors)
    predlist=torch.zeros(0,dtype=torch.long).to(device)
    lbllist=torch.zeros(0,dtype=torch.long).to(device)
    outlist = torch.zeros(0, dtype = torch.float).to(device)

    net.eval()
    end = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(VAL_LOADER):
        with torch.no_grad():
            # measure data loading time
            data_time.update(time.time() - end)
            
            inputs, targets = inputs.to(device), targets.to(device)

            inputs = inputs.contiguous().view(-1, 1, np.shape(inputs)[2], np.shape(inputs)[3])            
            
            outputs = net(inputs)
        
            outputs = outputs.reshape(int(model_args['batch_size']), 5, num_classes)
            outputs = torch.mean(outputs, 1)
            _, preds = torch.max(outputs, 1) 
            loss = criterion(outputs, targets)
            
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            err1 = 100. - prec1
            err5 = 100. - prec5

                
            losses.update(loss.item(), inputs.size(0))
            top1.update(err1.item(), inputs.size(0))
            top5.update(err5.item(), inputs.size(0))

            #Append batch prediction results
            outputs = outputs.unsqueeze(0)
            outlist=torch.cat([outlist,outputs], 0)
            predlist = torch.cat([predlist, preds.view(-1)], 0)
            lbllist=torch.cat([lbllist,targets.view(-1)], 0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    out = (' * Error@1 {top1.avg:.3f} Error@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    print(out)

    return losses.avg, predlist, lbllist, outlist


def get_dataloader(label_type):
    if label_type == 'verb':
        model_args = verb_args
    elif label_type == 'noun':
        model_args = noun_args
    print(model_args['label_type'])     
    image_transform = torchvision.transforms.Compose([
        transforms.ToTensor(),
    ])
    
    VAL = AudioDataset(args.data_path, 
                       evaluate_csv, 
                       sampling_rate = model_args['sampling_rate'], 
                       window_size = model_args['window_size'], 
                       step_size = model_args['hop_length'], 
                       n_fft = 512,
                       time_warp = model_args['time_warp'],
                       freq_mask = model_args['freq_mask'],
                       time_mask = model_args['time_mask'],
                       mask_size = model_args['mask_size'],
                       mask_num = model_args['mask_num'],
                       label_type = model_args['label_type'],
                       mode = 'val',
                       im_transform=image_transform)
    
    VAL_LOADER = DataLoader(dataset=VAL, 
                            batch_size=model_args['batch_size'], 
                            shuffle = False, 
                            drop_last = True, 
                            num_workers=28,
                            pin_memory=True)

    return VAL_LOADER
    
verb_model = models.resnet50()
with torch.no_grad():
    weights = torch.nn.Parameter(torch.mean(verb_model._modules['conv1'].weight, 1, True))
    verb_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    verb_model.conv1.weight = weights


noun_model = copy.deepcopy(verb_model)

num_ftrs = verb_model.fc.in_features
verb_model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(num_ftrs, num_verb_classes))
#verb_model = torch.nn.DataParallel(verb_model, device_ids = range(0, args.ngpus))

num_ftrs = noun_model.fc.in_features
noun_model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(num_ftrs, num_noun_classes))
noun_model = noun_model.to(device)
#noun_model = torch.nn.DataParallel(noun_model, device_ids = range(0, args.ngpus))

criterion = torch.nn.CrossEntropyLoss()

verb_predlist = None
noun_predlist = None
verb_lbllist = None
noun_lbllist = None
verb_outlist = None
noun_outlist = None

if args.verb_run is not None:
    verb_model_path = 'runs/EPIC_baselines/run_' + str(args.verb_run) + '/model.t7'
    verb_checkpoint = torch.load(verb_model_path)
    #print(verb_checkpoint['val_errors'])
    verb_model.load_state_dict(verb_checkpoint['state_dict'])
    verb_model = torch.nn.DataParallel(verb_model, device_ids = range(0, args.ngpus))
    dataloader = get_dataloader('verb')
    _, verb_predlist, verb_lbllist, verb_outlist = validate(verb_model, num_verb_classes, dataloader, verb_args)
if args.noun_run is not None:
    noun_model_path = 'runs/EPIC_baselines/run_' + str(args.noun_run) + '/model.t7'
    noun_checkpoint = torch.load(noun_model_path)
    noun_model.load_state_dict(noun_checkpoint['state_dict'])
    noun_model = torch.nn.DataParallel(noun_model, device_ids = range(0, args.ngpus))
    dataloader = get_dataloader('noun')
    _, noun_predlist, noun_lbllist, noun_outlist = validate(noun_model, num_noun_classes, dataloader, noun_args)



_  = create_csv(verb_lbllist, verb_predlist, verb_outlist, args.verb_run, noun_predlist, noun_lbllist, noun_outlist, args.noun_run) 

