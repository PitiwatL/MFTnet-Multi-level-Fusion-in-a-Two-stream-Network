import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import sklearn 
from torchvision import transforms
from PIL import Image
import torchvision.models as models

from Feeder.transform import train_transform


class VideoDataset(Dataset):
    def __init__(self, data_spatial,  data_optical, labels, transform = train_transform, 
                 num_sec_frame = 10, mode = 'Spatial'):
                 self.data_spatial = data_spatial
                 self.data_optical = data_optical
                 self.labels    = labels
                 self.transform = train_transform
                 self.mode      = mode
                 self.numsec    = num_sec_frame
        
    def __len__(self):
        if self.mode == 'Spatial' :
            return len(self.data_spatial)
            
        if self.mode == 'OpticalFlow' :
            return len(self.data_optical)
            
        if self.mode == 'IntermediateFusion' :
            return len(self.data_spatial)
    
    def __getitem__(self, idx):
        labels  = self.labels[idx]
        
        if self.mode == 'Spatial' :
            video_path = self.data_spatial[idx]
            x = SpatialGen(video_path, self.numsec)
            
            return x, torch.tensor(labels)
        
        if self.mode == 'OpticalFlow' :
            video_path = self.data_optical[idx]
            x = OpticalFlowGen(video_path, self.numsec)
            
            return x, torch.tensor(labels)
            
        if self.mode == 'IntermediateFusion' :
            video_path1 = self.data_spatial[idx]
            video_path2 = self.data_optical[idx]
            
            x1 = SpatialGen(video_path1, self.numsec)
            x2 = OpticalFlowGen(video_path2, self.numsec)
            
            return x1, x2, torch.tensor(labels)
        
    
def SpatialGen(video_path, num_sec_frame, transform = train_transform):
    cap = cv2.VideoCapture(video_path)
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for_every = num_frame//num_sec_frame
    
    VIDEO_out = torch.zeros(num_sec_frame, 3, 224, 224) 
    count = 0
    for idx in range(num_frame):
        ret, frame = cap.read()
        
        if (idx % for_every == 0) and (ret == True) : 
            frame = Image.fromarray(frame)
            frame = transform(frame)
            
            VIDEO_out[count, :, :, :] += frame
            
            count += 1
            if count == num_sec_frame: 
                break
    
    return VIDEO_out.detach().clone()
    
    
def OpticalFlowGen(video_path,  num_sec_frame, transform = train_transform):
    if num_sec_frame == 5:
        num_frame = [0, 5, 10, 15, 19]
    if num_sec_frame == 10:
        num_frame = [0, 2, 4, 6, 8, 10, 12, 14, 16, 19]
    if num_sec_frame == 15:
        num_frame = [0, 2, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    if num_sec_frame == 20:   
        num_frame = [idx for idx in range(20)]
    
    VIDEO_opt = torch.zeros(num_sec_frame, 3, 224, 224) 
    for idx, frame_idx in enumerate(num_frame): 
        frame = cv2.imread(video_path + "/" + str(idx) + ".jpg")
        frame = cv2.resize(frame, (224,224))
        
        GrayOpt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        GrayOpt = np.stack((GrayOpt,)*3, axis=-1)   

        Gopt_frame = Image.fromarray(GrayOpt.astype(np.uint8))
        Gopt_frame = transform(Gopt_frame)
        
        VIDEO_opt[idx, :, :, :] += Gopt_frame 
        
    return VIDEO_opt.detach().clone()
