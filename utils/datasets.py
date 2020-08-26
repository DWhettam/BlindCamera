from audio_records import EpicAudioRecord
from SpecAugment import spec_augment_pytorch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from utils.utils import *
from PIL import Image
import librosa
import numpy as np
import random
import pickle
import PIL
import os
import json
import torch
import torchaudio
import torchvision
import h5py
import datetime

class AudioDataset(Dataset):
    def __init__(self, audio_files, file_list, window_size, step_size, n_fft, time_warp, freq_mask, time_mask, mask_size, mask_num, augment = False, clip_len = 1.279, label_type = 'full',  sampling_rate = 24000, mode = 'train', im_transform = None):
        if not os.path.exists(audio_files) or not os.path.exists(file_list):
            raise Exception('path does not exist')
        self.hdf5_files = audio_files
        self.audio_files = None
        self.augment = augment
        self.clip_len = clip_len
        self.file_list = file_list
        self.im_transform = im_transform
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.step_size = step_size
        self.n_fft = n_fft
        self.time_warp = time_warp
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.mask_size = mask_size
        self.mask_num = mask_num
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
 
        if (end_seconds - start_seconds) > self.clip_len:
            if self.mode == 'val':                
                num_clips = 5
                clip_interval = int(round((((end_seconds - start_seconds) - self.clip_len) / num_clips) * self.sampling_rate))
                start = int(round(start_seconds * self.sampling_rate))
                for clip in range(num_clips):            
                    val_clip = data_point[start:(start+int(round((self.clip_len * self.sampling_rate))))]
                    data.append(val_clip)
                    start += clip_interval           
            else:
                start = int(round(start_seconds*self.sampling_rate))
                end = int(round((end_seconds- self.clip_len)*self.sampling_rate))
                rdm_strt = random.randint(start, end)
                rdm_end = rdm_strt + int(np.floor((self.clip_len*self.sampling_rate)))
                data.append(data_point[rdm_strt:rdm_end])
        else:
            mid_seconds = (start_seconds + end_seconds)/2
        
            left_seconds = mid_seconds - (self.clip_len / 2)
            right_seconds = mid_seconds + (self.clip_len / 2)
        
            left_sample = int(round(left_seconds * self.sampling_rate))
            right_sample = int(round(right_seconds * self.sampling_rate))
        
            duration = data_point.shape[0] / float(self.sampling_rate)
        
            if left_seconds < 0:
                data.append(data_point[:int(round(self.sampling_rate * self.clip_len))])
            elif right_seconds > duration:
                data.append(data_point[-int(round(self.sampling_rate * self.clip_len)):])
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
            if self.augment and self.mode == 'train':
                spec = spec_augment_pytorch.spec_augment(spec, 
                            time_warping_para = self.time_warp, frequency_masking_para = self.freq_mask, 
                            time_masking_para = self.time_mask, frequency_mask_num = self.mask_num, time_mask_num = self.mask_num)
            
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
