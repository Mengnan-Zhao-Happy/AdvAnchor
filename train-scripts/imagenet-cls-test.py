'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
import os
import pandas as pd
import torch
import cv2
import argparse
import torch.utils.data as data
import numpy as np
import torchvision.models as premodels
from torch.autograd import Variable
class ResNet50(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(ResNet50, self).__init__()
        if pretrained is True:
            self.model = premodels.resnet50(pretrained=True)
        else:
            self.model = premodels.resnet50(pretrained=False)
        # self.model.fc = nn.Linear(512, num_classes)
        # change the classification layer
        # self.l0 = nn.Linear(512, num_classes)
        # self.dropout = nn.Dropout2d(0.4)
    def forward(self, x):
        x = self.model(x) # 64 512 7 7 
        return x

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./evaluation_folder-diff/eval/0', type=str)
    parser.add_argument('--prompts_path', default='./data/imagenet_prompts.csv', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr_schedule', default='multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--normal_mean', default=0, type=float, help='normal_mean')
    parser.add_argument('--device', help='cuda device to train on', type=str, required=False, default='cuda:7')
    parser.add_argument('--normal_std', default=1, type=float, help='normal_std')
    return parser.parse_args()

# cassette player
# tape player
#  radio
def get_loaders(dir_, batch_size):
    data_transforms = {
        'eval': transforms.Compose([
            transforms.Resize(512),                    # [2]
            transforms.CenterCrop(512),                # [3]
            transforms.ToTensor(),                     # [4]
            transforms.Normalize(                      # [5]
                mean=[0.485, 0.456, 0.406],            # [6]
                std=[0.229, 0.224, 0.225])              # [7]
        ])}
    image_datasets = {x: datasets.ImageFolder(os.path.join(dir_, x), data_transforms[x])
                      for x in ['eval']}
    dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, pin_memory=False,
                                      num_workers=8)
                   for x in ['eval']}
    return dataloaders['eval']
def _one_hot(label,num_classes):
    one_hot = torch.eye(num_classes)[label]
    # result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(10 - 1))
    return one_hot
def logLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss
args = get_args()
df = pd.read_csv(args.prompts_path)
labels = []
for _, row in df.iterrows():
    labels.append(row.label_idx)
num_classes = 1000
# eval_loader = get_loaders(args.path, args.batch_size)

eval_acc = 0
eval_n = 0
model = ResNet50(True, num_classes)
model.to(args.device)
model.eval()
pos_ind = 0
transform_valid = transforms.Compose([
            transforms.Resize(256),                    # [2]
            transforms.CenterCrop(224),                # [3]
            transforms.ToTensor(),                     # [4]
            transforms.Normalize(                      # [5]
                mean=[0.485, 0.456, 0.406],            # [6]
                std=[0.229, 0.224, 0.225])              # [7]
                ])
for i in range(len(labels)):
    img_name = os.path.join(args.path, f'{i}_0.png')
    img = Image.open(img_name)
    img_ = transform_valid(img).unsqueeze(0)
    img_ = img_.to(args.device)
    output = model(img_)
    label = labels[i]
    if output.max(1)[1] == label:
        pos_ind += 1
print(len(labels),pos_ind, pos_ind/len(labels))