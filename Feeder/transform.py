import torch
from torchvision import transforms
from torchvision.transforms import ToTensor

train_transform = transforms.Compose([
        #transforms.RandomRotation(10),      # rotate +/- 10 degrees
        #transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize((224,224)),              ##### resize shortest side to 224 pixels
        #transforms.CenterCrop(224),         ##### crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.CenterCrop(224),          ##########################
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
