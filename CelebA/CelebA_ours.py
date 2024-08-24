#step 1 import image
import torchvision.datasets
import math
import torchvision.transforms as tvt
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wget
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from torchvision.utils import make_grid
from PIL import Image
from time import time
from tqdm import tqdm
import random
from transformers import ViTModel
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
from collections import OrderedDict




device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device_id = 1
image_size = 224
batch_size = 64
torch.set_num_threads(1)   # Sets the number of threads used for intra-operations
torch.set_num_interop_threads(1)   # Sets the number of threads used for inter-operations

def seed_everything(seed):
    """
    Changes the seed for reproducibility. 
    """
    random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def train_val_split(dataset, val_percent=0.15):
    val_size = int(len(dataset) * val_percent)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset




def get_transform_celebA(aug):
    if aug == False:
        transform=tvt.Compose([tvt.Resize((256,256)),
                               tvt.CenterCrop((224,224)),
                                tvt.ToTensor(),
                                tvt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                                  
                                ])
    if aug == True:
        transform=tvt.Compose([tvt.Resize((256,256)),
                               tvt.RandomResizedCrop(
                                    (224,224),
                                    scale=(0.7, 1.0),
                                    ratio=(0.75, 1.3333333333333333),
                                    interpolation=2),
                                tvt.RandomHorizontalFlip(),
                                tvt.ToTensor(),
                                tvt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                                  
                                ])
    return transform


seed_everything(0)

dr = "../../../Dataset"
dataset = torchvision.datasets.CelebA(dr,split='train', transform=get_transform_celebA(True))
test_dataset = torchvision.datasets.CelebA(dr,split='test', transform=get_transform_celebA(False))

# Splitting the dataset
train_dataset, val_dataset = train_val_split(dataset)

# Creating data loaders for training and validation
training_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
print("Training dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))
print('Done')  


def test(model, classifier, dataloader, print_fairness=True):
    model.eval()
    test_pred = []
    test_gt = []
    sense_gt = []
    female_predic = []
    female_gt = []
    male_predic = []
    male_gt = []
    correct_00, total_00 = 0, 0
    correct_01, total_01 = 0, 0
    correct_10, total_10 = 0, 0
    correct_11, total_11 = 0, 0
    
    # Evaluate on test set.
    for step, (test_input, attributes) in tqdm(enumerate(dataloader), total=len(dataloader), dynamic_ncols=False, ascii=True):
        sensitive, test_target = attributes[:,20], attributes[:,2]
        test_input = test_input.to(device)
        test_target = test_target.to(device)
    
        gt = test_target.detach().cpu().numpy()
        sen = sensitive.detach().cpu().numpy()
        test_gt.extend(gt)
        sense_gt.extend(sen)
    
        with torch.no_grad():
            test_feature = model(test_input)
           
            test_pred_  = classifier(test_feature)
            
            _, predic = torch.max(test_pred_.data, 1)
            predic = predic.detach().cpu()
            test_pred.extend(predic.numpy())
            label = test_target.squeeze().detach().cpu()
            mask_00 = ((label == 0) & (sensitive == 0))
            mask_01 = ((label == 0) & (sensitive == 1))
            mask_10 = ((label == 1) & (sensitive == 0))
            mask_11 = ((label == 1) & (sensitive == 1))
            
            correct_00 += (predic[mask_00] == label[mask_00]).float().sum().item()
            total_00 += mask_00.float().sum().item()
    
            correct_01 += (predic[mask_01] == label[mask_01]).float().sum().item()
            total_01 += mask_01.float().sum().item()
    
            correct_10 += (predic[mask_10] == label[mask_10]).float().sum().item()
            total_10 += mask_10.float().sum().item()
    
            correct_11 += (predic[mask_11] == label[mask_11]).float().sum().item()
            total_11 += mask_11.float().sum().item() 
    acc_00 = correct_00 / total_00
    acc_01 = correct_01 / total_01
    acc_10 = correct_10 / total_10
    acc_11 = correct_11 / total_11
    
    print(f'Accuracy for y=0, s=0: {acc_00}')
    print(f'Accuracy for y=0, s=1: {acc_01}')
    print(f'Accuracy for y=1, s=0: {acc_10}')
    print(f'Accuracy for y=1, s=1: {acc_11}')   
    wga = min(acc_00, acc_01, acc_10, acc_11)
    for i in range(len(sense_gt)):
        if sense_gt[i] == 0:
            female_predic.append(test_pred[i])
            female_gt.append(test_gt[i])
        else:
            male_predic.append(test_pred[i])
            male_gt.append(test_gt[i])
    female_CM = confusion_matrix(female_gt, female_predic)    
    male_CM = confusion_matrix(male_gt, male_predic) 
    female_dp = (female_CM[1][1]+female_CM[0][1])/(female_CM[0][0]+female_CM[0][1]+female_CM[1][0]+female_CM[1][1])
    male_dp = (male_CM[1][1]+male_CM[0][1])/(male_CM[0][0]+male_CM[0][1]+male_CM[1][0]+male_CM[1][1])
    female_TPR = female_CM[1][1]/(female_CM[1][1]+female_CM[1][0])
    male_TPR = male_CM[1][1]/(male_CM[1][1]+male_CM[1][0])
    female_FPR = female_CM[0][1]/(female_CM[0][1]+female_CM[0][0])
    male_FPR = male_CM[0][1]/(male_CM[0][1]+male_CM[0][0])
    if print_fairness == True:
        print('DP',abs(female_dp - male_dp))
        print('EOP', abs(female_TPR - male_TPR))
        print('EoD',0.5*(abs(female_FPR-male_FPR)+ abs(female_TPR-male_TPR)))
        print('acc', accuracy_score(test_gt, test_pred))
    return wga

def train_model():
    epoch =10
    weight_decay=1e-3
    init_lr=1e-3
    momentum_decay = 0.9
    schedule = False
    resnet50 = models.resnet50(pretrained=True)
    resnet50.fc = nn.Identity()
    num_classes = 2 
    classifier = nn.Linear(2048, num_classes)
    wg = 0
    model = resnet50.to(device)
    classifier = classifier.to(device)
    resnet50_parameters = model.parameters()
    classifier_parameters = classifier.parameters()
    combined_parameters = list(resnet50_parameters) + list(classifier_parameters)
    
    criterion = nn.CrossEntropyLoss()
    mean_criterion = nn.MSELoss()
    acc = 0
    optimizer = optim.SGD(combined_parameters, lr=init_lr, momentum=momentum_decay, weight_decay = weight_decay)
    optimizer_2 = optim.SGD(model.parameters(), lr=init_lr, momentum=momentum_decay, weight_decay = weight_decay)
    
    if schedule == True:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max= epoch)
    else:
        scheduler = None
    
    for epoches in range(epoch):
        with tqdm(training_data_loader, unit="batch", dynamic_ncols=False, ascii=True) as tepoch:
            model.train()
            feature_y_0_a0 = []
            feature_y_0_a1 = []
            feature_y_1_a0 = []
            feature_y_1_a1 = []
            loss00 = 0
            loss01 = 0
            loss10 = 0
            loss11 = 0

            with torch.no_grad(): 
                for step, (valid_input, valid_attributes) in enumerate(val_loader):
                    valid_target, validsensitive = valid_attributes[:,2], valid_attributes[:,20]
                    valid_input = valid_input.to(device)
                    with torch.no_grad():
                        valid_feature = model(valid_input)
                        label = valid_target.squeeze().detach().cpu()
                        mask_00 = ((label == 0) & (validsensitive == 0))
                        mask_01 = ((label == 0) & (validsensitive == 1))
                        mask_10 = ((label == 1) & (validsensitive == 0))
                        mask_11 = ((label == 1) & (validsensitive == 1))
                        g1 = valid_feature[mask_00]
                        g2 = valid_feature[mask_01]
                        g3 = valid_feature[mask_10]
                        g4 = valid_feature[mask_11]
                        feature_y_0_a0.extend(g1.detach().cpu().numpy())
                        feature_y_0_a1.extend(g2.detach().cpu().numpy())
                        feature_y_1_a0.extend(g3.detach().cpu().numpy())
                        feature_y_1_a1.extend(g4.detach().cpu().numpy())

                feature_g2 = np.array(feature_y_0_a1)
                feature_g3 = np.array(feature_y_1_a0)
                feature_g2_tensor = torch.from_numpy(feature_g2)
                feature_g3_tensor = torch.from_numpy(feature_g3)

                mu_1 = torch.mean(feature_g2_tensor, 0)
                mu_1 = mu_1 /torch.norm(mu_1)
                mu_2 = torch.mean(feature_g3_tensor, 0)
                mu_2 = mu_2 /torch.norm(mu_2)
                weight = torch.cat((mu_1.unsqueeze(0), mu_2.unsqueeze(0)), 0)
                with torch.no_grad():
                    classifier.weight = nn.Parameter(weight)
           
            for train_input, attribute in tepoch:
                train_target, sensitive = attribute[:, 2], attribute[:, 20]
                train_input = train_input.to(device)
                label = train_target.detach().cpu() 
                one_hot_labels = F.one_hot(train_target, num_classes=2)
                train_target = one_hot_labels.float().to(device)
                
                feature = model(train_input)
                classifier = classifier.to(device)
                outputs  = classifier(feature)             
                
                mask_00 = ((label== 0) & (sensitive == 0))
                mask_01 = ((label == 0) & (sensitive == 1))
                mask_10 = ((label == 1) & (sensitive == 0))
                mask_11 = ((label == 1) & (sensitive == 1))
                               
                count_00 = mask_00.sum()
                count_01 = mask_01.sum()
                count_10 = mask_10.sum()
                count_11 = mask_11.sum()
                
                if count_01==0 or count_10==0 or count_00 == 0 or count_11 == 0:
                    continue
                g1_f = feature[mask_00]
                g2_f = feature[mask_01]
                
                mu1 = torch.mean(g1_f, 0)
                mu2 = torch.mean(g2_f, 0)
                    
                g3_f = feature[mask_10]
                g4_f = feature[mask_11]
                
                mu3 = torch.mean(g3_f, 0)
                mu4 = torch.mean(g4_f, 0)
                
                loss_mean = mean_criterion(mu3, mu4) + mean_criterion(mu1, mu2)
                
                if count_00 > 0:
                    loss_00 = criterion(outputs[mask_00], train_target[mask_00])
                    loss00 += loss_00.item()
                else:
                    loss_00 = torch.tensor(0)
                if count_01 > 0:
                    loss_01 = criterion(outputs[mask_01], train_target[mask_01])
                    loss01 += loss_01.item()
                else:
                    loss_01 = torch.tensor(0)
                if count_10 > 0:
                    loss_10 = criterion(outputs[mask_10], train_target[mask_10])
                    loss10 += loss_10.item()
                else:
                    loss_10 = torch.tensor(0)
                if count_11 > 0:
                    loss_11 = criterion(outputs[mask_11], train_target[mask_11])
                    loss11 += loss_11.item()
                else:
                    loss_11 = torch.tensor(0)

                loss = loss_00 + loss_01 + loss_10 + loss_11 + loss_mean
                
                
                tepoch.set_postfix(ut_loss = loss.item()) 
                optimizer_2.zero_grad()    
                loss.backward()
                optimizer_2.step()
                tepoch.set_description(f"epoch %2f " % epoches)
            if schedule:
                scheduler.step()
                       
        print("loss g1 (label=0, sensitive=0):",loss00 )
        print("loss g2 (label=0, sensitive=1):",loss01 )
        print("loss g3 (label=1, sensitive=0):",loss10 )
        print("loss g4 (label=1, sensitive=1):",loss11 )
        print('mean loss:', loss_mean.item())
        wga =test(model, classifier, test_data_loader, print_fairness=False)
        if wga > wg:
            wg = wga
            torch.save(model, 'CelebA_model.pth')
            torch.save(classifier, 'classifier.pth')
            
        




train_model()
print('**********************************')
print('************Evaluation************')
resnet50 = models.resnet50(pretrained=False)
resnet50.fc = nn.Identity()
num_classes = 2 
classifier = nn.Linear(2048, num_classes)
classifier = torch.load('classifier.pth')
model = resnet50
model = torch.load('CelebA_model.pth')
model = model.to(device)
classifier = classifier.to(device)
model.eval()
classifier.eval()
_  = test(model, classifier, test_data_loader, True)