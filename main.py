import numpy as np
import os, shutil
import torch
import torch.nn as nn
import torchvision.models as models

import sklearn
import tqdm 
from tqdm import tqdm
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

################### Import Tools ##########################################
from Feeder.transform import train_transform, val_transform
from models.Streams import OpticalFlowStream, SpatialStream, IntermediateFusion
from Feeder.datagen import datagen

from utils.callbacks import callbacks
############################# Device ########################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.cuda.is_available()
print("GPU Available: ", x)
print('Num Avil GPUs: ', torch.cuda.device_count())
#############################################################################
SaveWeight_Path     = '******** Specified Where To save weight ***********'
DATASETPATH_Spatial = '******** Datapath for Spatial Data **********'
DATASETPATH_OpticalFlow = '******** Datapath for OpticalFlow Data ********** '
num_sec_frame = 10          # 5, 10, 15, 20
MODE = 'IntermediateFusion' # MODE: Spatial, Optical Flow, or IntermediateFusion 
ratio = 0.8

train_loader, test_loader = datagen(DATASETPATH_Spatial, 
                                    DATASETPATH_OpticalFlow,      
                                    MODE,
                                    num_sec_frame,    
                                    ratio)

# model = OpticalFlowStream(512, dropout_rate = 0.2)

# IntermediateFusion : SumFusion, MaxFusion, ConcatenationFusion, ConvolutionFusion
model = IntermediateFusion(512, Method = 'ConvolutionFusion', dropout_rate = 0.2)
model = model.to(device)
model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr= 0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, 
                                     patience=3, min_lr= 1 * 1e-6, verbose = True)

########################
###### load Weight #####
########################
#model.load_state_dict(torch.load('/project/lt200048-video/Plueangw/AttentionMMnet/Weight/VGGAttn_4_20.pt'))

### Start Training Loop
epochs = 200

val_loss_his = []
train_loss_his = []
count = 0
for eph in range(epochs):
    loss_epoch_train = []
    loss_epoch_val = []
    
    Train_Correct = 0
    total_train = 0
    model.train()
    for b, (X_trainSpatial, X_trainOptical, y_train) in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        
        X_trainSpatial, X_trainOptical = X_trainSpatial.to(device), X_trainOptical.to(device)
        y_train = y_train.to(device)
        
        output = model(X_trainSpatial, X_trainOptical)
        y_pred = output
        y_prd  = torch.argmax(y_pred, 1)
        Train_Correct += torch.sum(y_prd == y_train)
        
        total_train += y_train.shape[0]
        loss = criterion(y_pred.cpu(), y_train.cpu())
        
        loss_epoch_train.append(loss.item())
                
        loss.backward()
        optimizer.step()
        
    
    train_loss_his.append(np.mean(loss_epoch_train))
    train_acc = Train_Correct/total_train
    print(f'epoch: {eph:2}  Train loss: {np.mean(loss_epoch_train):10.7f} : Train Acc {train_acc:10.7f}')
    
    # Run the validation batches
    Val_Correct = 0
    total_test = 0
    model.eval()
    with torch.no_grad():
        for b, (X_testSpatial, X_testOptical, y_val) in enumerate(test_loader):
            X_trainSpatial, X_trainOptical = X_trainSpatial.to(device), X_trainOptical.to(device)
            y_val = y_val.to(device)
            
            out_val = model(X_testSpatial, X_testOptical)
            
            y_val_ = torch.argmax(out_val, 1)
            Val_Correct += torch.sum(y_val_ == y_val)
            
            total_test += y_val.shape[0]
            loss = criterion(out_val.cpu(), y_val.cpu())
            
            loss_epoch_val.append(loss.item())
            
    val_acc = Val_Correct/total_test
    val_loss_his.append(np.mean(loss_epoch_val))
    scheduler.step(np.mean(loss_epoch_val))
    
    print(f'Epoch: {eph} Val Loss: {np.mean(np.array(loss_epoch_val, dtype=np.float32)):10.7f} \
          : Val Acc {val_acc:10.7f}')

    RES = callbacks(eph, model, loss_epoch_val, val_loss_his, count,
                    SaveWeight_path = SaveWeight_Path)
    
    if RES == "continue": 
        count = 0
    if RES == "not_improve": 
        count += 1
        if count == 20:
            break
    if RES == "Break": 
        break
    