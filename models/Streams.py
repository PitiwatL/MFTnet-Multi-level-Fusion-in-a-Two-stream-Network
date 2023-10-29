from .LSTM import LSTMCells, ConvLstm
import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################## Define Pretrained Models ###############
vgg16 = models.vgg16_bn(pretrained = True)
vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, 1000)
VggFirstHalf = vgg16.features[0:21].to(device) 

Densenet121 = models.densenet121(pretrained = True)
Densenet121.classifier = nn.Linear(Densenet121.classifier.in_features, 512)
DensePart = nn.Sequential(Densenet121.features[0:4], 
                          Densenet121.features.denseblock1).to(device) 

VggSecondHalf1 = nn.Sequential(vgg16.features[21:],
                              vgg16.avgpool).to(device)
VggSecondHalf2 = vgg16.classifier.to(device)

############################################################

class OpticalFlowStream(nn.Module):
    def __init__(self, hidden_unit, dropout_rate = 0.2):
        super(OpticalFlowStream, self).__init__()
        self.hidden_dim = hidden_unit
        self.cf1 = nn.Linear(hidden_unit, 256)
        self.cf2 = nn.Linear(256, 128)
        self.cf3 = nn.Linear(128, 101)
        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu    = nn.ReLU()

        self.ConvLSTM1 = ConvLstm(512, hidden_unit, pretrained = "DenseNet121") # DenseNet121 or VGG16

        # self.batchnorm1 = nn.BatchNorm1d(10, 256)
        self.batchnorm2 = nn.BatchNorm1d(hidden_unit)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.batchnorm4 = nn.BatchNorm1d(101)

        # self.double()

    def forward(self, Seq):
        x  = self.ConvLSTM1(Seq)
        x  = self.batchnorm2(x)
        x  = self.dropout(x)
        
        x  = self.relu(self.batchnorm3(self.cf1(x)))
        x  = self.dropout(x)
        
        x  = self.relu(self.cf2(x))
        x  = self.cf3(x)
       
        return x
    
class SpatialStream(nn.Module):
    def __init__(self, hidden_unit, dropout_rate = 0.2):
        super(SpatialStream, self).__init__()
        self.hidden_dim = hidden_unit
        self.cf1 = nn.Linear(hidden_unit, 256)
        self.cf2 = nn.Linear(256, 128)
        self.cf3 = nn.Linear(128, 101)
        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu    = nn.ReLU()

        self.ConvLSTM1 = ConvLstm(1000, hidden_unit, pretrained = "VGG16") # DenseNet121 or VGG16

        # self.batchnorm1 = nn.BatchNorm1d(10, 256)
        self.batchnorm2 = nn.BatchNorm1d(hidden_unit)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.batchnorm4 = nn.BatchNorm1d(101)

        # self.double()

    def forward(self, Seq):
        x  = self.ConvLSTM1(Seq)
        x  = self.batchnorm2(x)
        x  = self.dropout(x)
        
        x  = self.relu(self.batchnorm3(self.cf1(x)))
        x  = self.dropout(x)
        
        x  = self.relu(self.cf2(x))
        x  = self.cf3(x)
       
        return x
    

class IntermediateFusion(nn.Module):
    def __init__(self, hidden_unit, Method = 'SumFusion', dropout_rate = 0.2):
        super(IntermediateFusion, self).__init__()
        
        self.hidden_dim = hidden_unit
        self.cf1 = nn.Linear(hidden_unit, 256)
        self.cf2 = nn.Linear(256, 128)
        self.cf3 = nn.Linear(128, 101)
        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu    = nn.ReLU()
        
        self.DensePart     = DensePart      # [batch, 256, 56, 56]
        self.VGGFirstHalf  = VggFirstHalf   # [batch, 256, 56, 56]
        self.VGGSecondHalf1 = VggSecondHalf1
        self.VGGSecondHalf2 = VggSecondHalf2
        self.ConvFusion     = nn.Conv2d(512, 256, (1, 1))
        
        # self.batchnorm1 = nn.BatchNorm1d(10, 256)
        self.batchnorm2 = nn.BatchNorm1d(hidden_unit)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.batchnorm4 = nn.BatchNorm1d(101)
        
        self.method     = Method
        
        self.cell1 = LSTMCells(input_fea = 1000, hidden_unit = self.hidden_dim)
        self.cell2 = LSTMCells(input_fea = self.hidden_dim, hidden_unit = self.hidden_dim)

        # self.double()

    def forward(self, SeqSpatial, SeqOptical):
        ct  = torch.zeros(SeqSpatial.shape[0], self.hidden_dim).to(device) 
        ht  = torch.zeros(SeqSpatial.shape[0], self.hidden_dim).to(device)
        
        return_seq = torch.zeros(SeqSpatial.shape[0], SeqSpatial.shape[1], self.hidden_dim).to(device)
        for t in range(SeqSpatial.shape[1]): # [B, F, 3, 224, 224]
            ext1 = self.DensePart(SeqOptical[:, t, :, :, :])   # [Batch, Extract_Features_Dim]
            ext2 = self.VGGFirstHalf(SeqSpatial[:, t, :, :, :])
            
            if self.method == 'SumFusion': # [8, 256, 56, 56]
                Fusion_ext = torch.add(ext1, ext2)
            if self.method == 'MaxFusion':
                Fusion_ext = torch.max(ext1, ext2)
            if self.method == 'ConcatenationFusion':
                Fusion_ext = torch.cat((ext1, ext2), 1)
            if self.method == 'ConvolutionFusion':
                Fusion_ext = torch.cat((ext1, ext2), 1)
                Fusion_ext = self.ConvFusion(Fusion_ext)
                
            blendfeature = self.VGGSecondHalf1(Fusion_ext)  # [batch, 512, 7, 7]
            blendfeature = torch.flatten(blendfeature, start_dim=1, end_dim=-1) # [batch, 25088]
            blendfeature = self.VGGSecondHalf2(blendfeature)

            ct, ht = self.cell1(blendfeature, ct, ht)
            
            return_seq[:, t, :] += ht 
            
        ct_ = torch.zeros(SeqSpatial.shape[0], self.hidden_dim).to(device) 
        ht_ = torch.zeros(SeqSpatial.shape[0], self.hidden_dim).to(device)
        for t in range(SeqSpatial.shape[1]):
            ct_, ht_ = self.cell2(return_seq[:, t, :], ct_, ht_)
        
        x  = self.batchnorm2(ht_)
        x  = self.dropout(x)
        
        x  = self.relu(self.batchnorm3(self.cf1(x)))
        x  = self.dropout(x)
        
        x  = self.relu(self.cf2(x))
        x  = self.cf3(x)
        
        return x
