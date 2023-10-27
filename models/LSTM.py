import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMCells(nn.Module):
    def __init__(self, input_fea, hidden_unit, recurrent_dropout = None):
        super(LSTMCells, self).__init__()
        self.hidden_dim = hidden_unit
        self.input_fea  = input_fea

        self.Linear1 = nn.Linear(input_fea + hidden_unit, hidden_unit, bias=True)
        self.Linear2 = nn.Linear(input_fea + hidden_unit, hidden_unit, bias=True)
        self.Linear3 = nn.Linear(input_fea + hidden_unit, hidden_unit, bias=True)
        self.Linear4 = nn.Linear(input_fea + hidden_unit, hidden_unit, bias=True)

        self.Linear5 = nn.Linear(hidden_unit, hidden_unit, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()

    def forward(self, x, ct, ht): # [batch, Num_Cells, Depth] --> [batch, 1, Depth] 
        Input1 = torch.cat((ht, x), -1).float().to(device)

        sigma1 = self.sigmoid(self.Linear1(Input1))
        sigma2 = self.sigmoid(self.Linear2(Input1))
        sigma3 = self.sigmoid(self.Linear3(Input1))

        mul1 = sigma1 * ct
        mul2 = sigma2 * self.tanh(self.Linear4(Input1))

        c_next = mul1 + mul2

        # h_next = self.tanh(self.Linear5(ct)) * sigma3
        h_next = self.tanh(ct) * sigma3

        return c_next, h_next

        if self.return_sequence == False : return h_next
        if self.return_sequence == True  : return Seq
        
##### Define the pretrain model ###################################
# vgg16 = models.vgg16_bn()
# vgg16.load_state_dict(torch.load("/tarafs/data/project/proj0173-action/PlueangwMiniMLFnet/pretrained_weight/vgg16_bn-6c64b313.pth"))
# vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, 256)
# vgg16_ = vgg16.to(device) # Out 256
####################################################################

# for param in vgg16_.parameters():
#     param.requires_grad = True
    
class ConvLstm(nn.Module):
      def __init__(self, input_fea, hidden_unit, return_sequence = False, pretrained = ''):
        super(ConvLstm, self).__init__()
        vgg16 = models.vgg16_bn()
        vgg16.load_state_dict(pretrained = True)
        vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, 1000)
        vgg16_ = vgg16.to(device) # Out 256
        
        Densenet121 = models.densenet121()
        Densenet121.load_state_dict(pretrained = True)
        Densenet121.classifier = nn.Linear(Densenet121.classifier.in_features, 512)
        densenet121_ = Densenet121.to(device) # Out 256
        

        self.hidden_unit = hidden_unit
        self.return_sequence = return_sequence
        self.cell1 = LSTMCells(input_fea = input_fea, hidden_unit = self.hidden_unit)
        self.cell2 = LSTMCells(input_fea = self.hidden_unit, hidden_unit = self.hidden_unit)
        self.pretrained = pretrained
        
        if self.pretrained == "VGG16":
            print("VGG16")
            self.Pretrained = vgg16_
        if self.pretrained == "DenseNet121":
            print("DenseNet121")
            self.Pretrained = densenet121_
        
      
      def forward(self, Seq): # [batch, Num_Cells, Depth]
        ct  = torch.zeros(Seq.shape[0], self.hidden_unit).to(device) 
        ht  = torch.zeros(Seq.shape[0], self.hidden_unit).to(device)
        
        return_seq = torch.zeros(Seq.shape[0], Seq.shape[1], self.hidden_unit).to(device)
        for t in range(Seq.shape[1]): # [B, F, 3, 224, 224]
            ext = self.Pretrained(Seq[:, t, :, :, :])   # [Batch, Extract_Features_Dim]
            
            ct, ht = self.cell1(ext, ct, ht)
            
            return_seq[:, t, :] += ht 
        
        ct_ = torch.zeros(Seq.shape[0], self.hidden_unit).to(device) 
        ht_ = torch.zeros(Seq.shape[0], self.hidden_unit).to(device)
        for t in range(Seq.shape[1]):
            ct_, ht_ = self.cell2(return_seq[:, t, :], ct_, ht_)
          
        # if self.return_sequence == False :    
            
        return ht_ 
    