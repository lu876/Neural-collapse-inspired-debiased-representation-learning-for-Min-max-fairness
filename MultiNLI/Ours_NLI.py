# New Version for any tokenizer

import os
import sys
import numpy as np
import random
import string
import pandas as pd
from transformers import BertTokenizer
import time
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from transformers import BertModel, AlbertModel
import transformers

transformers.logging.set_verbosity_error()



def seed_everything(seed):
    """
    Changes the seed for reproducibility. 
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device('cuda:2')
device_id = 2

#################important########################
seed_everything(2048)
start = time.time()
############## Select your model#########################
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
################ Paths and other configs - Set these #################################

data_dir = '../../../Dataset/MultiNLI'
glue_dir = '../../../Dataset/MultiNLI'

type_of_split = 'random'
assert type_of_split in ['preset', 'random']

######################################################################################

def tokenize(s):
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = s.lower()
    s = s.split(' ')
    return s

### Read in data and assign train/val/test splits
train_df = pd.read_json(
    os.path.join(
        data_dir,
        'multinli_1.0_train.jsonl'),
    lines=True)

val_df = pd.read_json(
    os.path.join(
        data_dir,
        'multinli_1.0_dev_matched.jsonl'),
    lines=True)

test_df = pd.read_json(
    os.path.join(
        data_dir,
        'multinli_1.0_dev_mismatched.jsonl'),
    lines=True)

split_dict = {
    'train': 0,
    'val': 1,
    'test': 2
}

if type_of_split == 'preset':
    train_df['split'] = split_dict['train']
    val_df['split'] = split_dict['val']
    test_df['split'] = split_dict['test']
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

elif type_of_split == 'random':
    np.random.seed(42)
    val_frac = 0.2
    test_frac = 0.3

    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    n = len(df)
    n_val = int(val_frac * n)
    n_test = int(test_frac * n)
    n_train = n - n_val - n_test
    splits = np.array([split_dict['train']] * n_train + [split_dict['val']] * n_val + [split_dict['test']] * n_test)
    np.random.shuffle(splits)
    df['split'] = splits
    
    
df = df.loc[df['gold_label'] != '-', :]
print(f'Total number of examples: {len(df)}')
for k, v in split_dict.items():
    print(k, np.mean(df['split'] == v))

label_dict = {
    'contradiction': 0,
    'entailment': 1,
    'neutral': 2
}
for k, v in label_dict.items():
    idx = df.loc[:, 'gold_label'] == k
    df.loc[idx, 'gold_label'] = v   
    
### Assign spurious attribute (negation words)
negation_words = ['nobody', 'no', 'never', 'nothing'] # Taken from https://arxiv.org/pdf/1803.02324.pdf

df['sentence2_has_negation'] = [False] * len(df)
for negation_word in negation_words:
    df['sentence2_has_negation'] |= [negation_word in tokenize(sentence) for sentence in df['sentence2']]

    
df['sentence2_has_negation'] = df['sentence2_has_negation'].astype(int) ##this is a


s1= df['sentence1'].str.split()
max_1 = s1.apply(lambda x: len(max(x, key=len)))

s2= df['sentence2'].str.split()
max_2 = s2.apply(lambda x: len(max(x, key=len)))
print(max_1.max(), max_2.max())

#df['tokenized_data'] = df.apply(lambda row: tokenizer.encode_plus(row['sentence1'], row['sentence2'], return_tensors='pt', padding = 'max_length'), axis = 1)
df['tokenized_data'] = df.apply(lambda row: tokenizer.encode_plus(row['sentence1'], row['sentence2'], return_tensors='pt', padding = 'max_length', max_length=128, truncation=True), axis = 1)
end = time.time()
print('elapse:', end-start)



class MultiNLIDataset():

    def __init__(self, dataframe, split_type):
        # Ensure the split type is valid
        if split_type not in ['train', 'val', 'test']:
            raise ValueError("Invalid split type. Must be one of 'train', 'val', 'test'")
        
        # Filter dataframe based on split_type
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        dataframe = dataframe[dataframe['split'] == self.split_dict[split_type]]
        dataframe.reset_index(drop=True, inplace=True)

        # Get the y values (and others) from the filtered dataframe
        self.y_array = dataframe['gold_label'].values
        self.confounder_array = dataframe['sentence2_has_negation'].values
        self.features_array = dataframe['tokenized_data']

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        a = self.confounder_array[idx]
        x = self.features_array[idx]
        return x, y, a

training_set = MultiNLIDataset(df, 'train')
vali_set = MultiNLIDataset(df, 'val')
test_set = MultiNLIDataset(df, 'test')
batch_size = 32
training_data_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, drop_last=True)      
valid_data_loader = torch.utils.data.DataLoader(vali_set, batch_size=batch_size, shuffle=True, drop_last=False)      
test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)      
print('Done')




class BERT_base(nn.Module):
    def __init__(self, BERT, type_id):
        super(BERT_base, self).__init__()
        if type_id == 'base' or type_id == 'small':
            p_dim = 768
        if type_id == 'large':
            p_dim = 1024
        self.BERT = BERT
        

    def forward(self, x):
        x['input_ids'] = x['input_ids'].squeeze(1)
        x['token_type_ids'] = x['token_type_ids'].squeeze(1)
        x['attention_mask'] = x['attention_mask'].squeeze(1)
        x = self.BERT(**x)
        x = x.last_hidden_state
        y = x[:,0]
        return y


# In[18]:


import torch.nn.functional as F
def compute_fairness(cf1, cf2):
    dp = []
    TPR = []
    FPR = []
    for cf in (cf1, cf2):
        TP = np.diag(cf)
        FN = cf.sum(axis =1)-np.diag(cf)
        FP = cf.sum(axis = 0) - np.diag(cf)
        TN = cf.sum()-(FN+FP+TP)

        dp_value = (TP+FP)/(TN+FP+FN+TP)
        TPR_value = TP/(TP+FN)
        FPR_value = FP/(FP+TN)
        dp.append(dp_value)
        TPR.append(TPR_value)
        FPR.append(FPR_value)
    DP = abs(dp[0]-dp[1])
    EoP = abs(TPR[0] - TPR[1])
    EoD = 0.5*(abs(FPR[0]-FPR[1])+abs(TPR[0]-TPR[1]))
    return DP, EoP, EoD  


def test(model, classifier, dataloader, print_fairness=True):
    model.eval()
    classifier = classifier.to(device)
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
    correct_20, total_20 = 0, 0
    correct_21, total_21 = 0, 0
    for step, (test_word, test_target, test_sensitive) in enumerate(dataloader):
        test_word = test_word.to(device)
        test_target = test_target.to(device)
        sensitive = test_sensitive
        label = test_target.squeeze().detach().cpu()
        gt = test_target.detach().cpu().numpy()
        sen = test_sensitive.detach().cpu().numpy()
        test_gt.extend(gt)
        sense_gt.extend(sen)

        with torch.no_grad():
            test_feature = model(test_word)
            prediction  = classifier(test_feature)
            test_pred_ = torch.argmax(prediction.data, dim=1)
            test_pred.extend(test_pred_.detach().cpu().numpy())
            predic =  test_pred_.detach().cpu()
            
            mask_00 = ((label == 0) & (sensitive == 0))
            mask_01 = ((label == 0) & (sensitive == 1))
            mask_10 = ((label == 1) & (sensitive == 0))
            mask_11 = ((label == 1) & (sensitive == 1))
            mask_20 = ((label == 2) & (sensitive == 0))
            mask_21 = ((label == 2) & (sensitive == 1))
            correct_00 += (predic[mask_00] == label[mask_00]).float().sum().item()
            total_00 += mask_00.float().sum().item()

            correct_01 += (predic[mask_01] == label[mask_01]).float().sum().item()
            total_01 += mask_01.float().sum().item()

            correct_10 += (predic[mask_10] == label[mask_10]).float().sum().item()
            total_10 += mask_10.float().sum().item()

            correct_11 += (predic[mask_11] == label[mask_11]).float().sum().item()
            total_11 += mask_11.float().sum().item() 
            
            correct_20 += (predic[mask_20] == label[mask_20]).float().sum().item()
            total_20 += mask_20.float().sum().item()

            correct_21 += (predic[mask_21] == label[mask_21]).float().sum().item()
            total_21 += mask_21.float().sum().item() 
                
                
    acc_00 = correct_00 / total_00
    acc_01 = correct_01 / total_01
    acc_10 = correct_10 / total_10
    acc_11 = correct_11 / total_11
    acc_20 = correct_20 / total_20
    acc_21 = correct_21 / total_21

    print(f'Accuracy for y=0, s=0: {acc_00}')
    print(f'Accuracy for y=0, s=1: {acc_01}')
    print(f'Accuracy for y=1, s=0: {acc_10}')
    print(f'Accuracy for y=1, s=1: {acc_11}')   
    print(f'Accuracy for y=2, s=0: {acc_20}')
    print(f'Accuracy for y=2, s=1: {acc_21}')
    wga = min(acc_00, acc_01, acc_10, acc_11, acc_20, acc_21)
    for i in range(len(sense_gt)):
        if sense_gt[i] == 0:
            female_predic.append(test_pred[i])
            female_gt.append(test_gt[i])
        else:
            male_predic.append(test_pred[i])
            male_gt.append(test_gt[i])
    female_CM = confusion_matrix(female_gt, female_predic) 
    male_CM = confusion_matrix(male_gt, male_predic) 
    DP, EoP, EoD = compute_fairness(female_CM , male_CM)
    ACC = accuracy_score(test_gt, test_pred)
    print('acc:', ACC)
    if print_fairness == True:            
        print('DP:', max(DP))
        print('EoP' , max(EoP))
        print('EoD', max(EoD))
    return wga
    



#x,y,a
def train_model(type_id):
    epoch = 5
    if type_id == 'base':
        BERT = BertModel.from_pretrained("bert-base-uncased")
        lr = 2e-5
    if type_id =='large':
        BERT = BertModel.from_pretrained("bert-large-uncased")
        lr = 1e-5
    if type_id =='small':
        BERT = AlbertModel.from_pretrained("textattack/albert-base-v2-imdb")
        lr = 1e-5
        
    model = BERT_base(BERT, type_id).to(device)                
    classifier = nn.Linear(768, 3)  
    classifier = classifier.to(device)
    
    #weight_decay=1e-5
    criterion = nn.CrossEntropyLoss()
    mean_criterion = nn.MSELoss()
    acc = 0
    wg = 0
    model_parameters = model.parameters()
    classifier_parameters = classifier.parameters()
    combined_parameters = list(model_parameters) + list(classifier_parameters)
    
    optimizer = optim.AdamW(combined_parameters, lr=lr)
    optimizer_2 = optim.AdamW(model.parameters(), lr=lr)
    
    
    for epoches in range(1):
        with tqdm(training_data_loader, unit="batch", dynamic_ncols=False, ascii=True) as tepoch:
            model.train()
            for word, train_target, _ in tepoch:
                word = word.to(device)
                train_target = torch.nn.functional.one_hot(train_target, num_classes=3)
                train_target = train_target.float().to(device)
                optimizer.zero_grad()
                feature = model(word)
                outputs  = classifier(feature)
                loss = criterion(outputs, train_target)
                tepoch.set_postfix(ut_loss = loss.item())   
                loss.backward()
                optimizer.step()
                tepoch.set_description(f"epoch %2f " % epoches)
    
    for epoches in range(epoch):
        
        with tqdm(training_data_loader, unit="batch", dynamic_ncols=False, ascii=True) as tepoch:
            model.train()
            feature_y_0_a0 = []
            feature_y_0_a1 = []
            feature_y_1_a0 = []
            feature_y_1_a1 = []
            feature_y_2_a0 = []
            feature_y_2_a1 = []
            loss00 = 0
            loss01 = 0
            loss10 = 0
            loss11 = 0
            loss20 = 0
            loss21 = 0
            
            
            with torch.no_grad(): 
                for step, (valid_input, valid_target, validsensitive) in enumerate(valid_data_loader):
                    valid_input = valid_input.to(device)
                    valid_feature = model(valid_input)
                    label = valid_target.squeeze().detach().cpu()
                    mask_00 = ((label == 0) & (validsensitive == 0))
                    mask_01 = ((label == 0) & (validsensitive == 1))
                    mask_10 = ((label == 1) & (validsensitive == 0))
                    mask_11 = ((label == 1) & (validsensitive == 1))
                    mask_20 = ((label == 2) & (validsensitive == 0))
                    mask_21 = ((label == 2) & (validsensitive == 1))
                    
                    g1 = valid_feature[mask_00]
                    g2 = valid_feature[mask_01]
                    g3 = valid_feature[mask_10]
                    g4 = valid_feature[mask_11]
                    g5 = valid_feature[mask_20]
                    g6 = valid_feature[mask_21]
                    
                    
                    feature_y_0_a0.extend(g1.detach().cpu().numpy())
                    feature_y_0_a1.extend(g2.detach().cpu().numpy())
                    feature_y_1_a0.extend(g3.detach().cpu().numpy())
                    feature_y_1_a1.extend(g4.detach().cpu().numpy())
                    feature_y_2_a0.extend(g5.detach().cpu().numpy())
                    feature_y_2_a1.extend(g6.detach().cpu().numpy())
                    
        

                feature_g1 = np.array(feature_y_0_a0)
                feature_g3 = np.array(feature_y_1_a0)
                feature_g5 = np.array(feature_y_2_a0)
                
                feature_g1_tensor = torch.from_numpy(feature_g1)
                feature_g3_tensor = torch.from_numpy(feature_g3)
                feature_g5_tensor = torch.from_numpy(feature_g5)

                mu_1 = torch.mean(feature_g1_tensor, 0)
                mu_1 = mu_1 /torch.norm(mu_1)
                mu_2 = torch.mean(feature_g3_tensor, 0)
                mu_2 = mu_2 /torch.norm(mu_2)
                mu_3 = torch.mean(feature_g5_tensor, 0)
                mu_3 = mu_3 /torch.norm(mu_3)
                weight = torch.cat((mu_1.unsqueeze(0), mu_2.unsqueeze(0), mu_3.unsqueeze(0)), 0)
                print(weight,"sim:",  F.cosine_similarity(mu_1.unsqueeze(0), mu_2.unsqueeze(0)) )

                with torch.no_grad():
                    classifier.weight = nn.Parameter(weight)

            for word, train_target, sensitive in tepoch:
                classifier = classifier.to(device)
                word = word.to(device)
                label = train_target.squeeze().detach().cpu()
                one_hot_labels = torch.nn.functional.one_hot(train_target, num_classes=3)
                train_target = one_hot_labels.float().to(device)
                
                feature = model(word)
                
                outputs  = classifier(feature)
    
                mask_00 = ((label== 0) & (sensitive == 0))
                mask_01 = ((label == 0) & (sensitive == 1))
                mask_10 = ((label == 1) & (sensitive == 0))
                mask_11 = ((label == 1) & (sensitive == 1))
                mask_20 = ((label == 2) & (sensitive == 0))
                mask_21 = ((label == 2) & (sensitive == 1))
                
                
                
                count_00 = mask_00.sum()
                count_01 = mask_01.sum()
                count_10 = mask_10.sum()
                count_11 = mask_11.sum()
                count_20 = mask_20.sum()
                count_21 = mask_21.sum()
                
                if count_21 == 0 or count_11 == 0:
                    continue
                
                
                g1_f = feature[mask_00]
                g2_f = feature[mask_01]
                
                mu1 = torch.mean(g1_f, 0)
                mu2 = torch.mean(g2_f, 0)
                    
                g3_f = feature[mask_10]
                g4_f = feature[mask_11]
                
                mu3 = torch.mean(g3_f, 0)
                mu4 = torch.mean(g4_f, 0)
                
                g5_f = feature[mask_20]
                g6_f = feature[mask_21]
                
                mu5 = torch.mean(g5_f, 0)
                mu6 = torch.mean(g6_f, 0)
                
                
                if count_00 > 0 and count_01 >0:
                    l1 = mean_criterion(mu1, mu2)
                else:
                    l1 = torch.tensor(0)
                    
                if count_10 > 0 and count_11 >0:
                    l2 = mean_criterion(mu3, mu4)
                else:
                    l2 = torch.tensor(0)
                if count_20 > 0 and count_21 >0:
                    l3 = mean_criterion(mu5, mu6)
                else:
                    l3 = torch.tensor(0)
                    
                    
                loss_mean = l1+l2+ l3
                
                
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
                    
                if count_20 > 0:
                    loss_20 = criterion(outputs[mask_20], train_target[mask_20])
                    loss20 += loss_20.item()
                else:
                    loss_20 = torch.tensor(0)
                if count_21 > 0:
                    loss_21 = criterion(outputs[mask_21], train_target[mask_21])
                    loss21 += loss_21.item()
                else:
                    loss_21 = torch.tensor(0)

                loss = loss_00 + loss_01 + loss_10 + loss_11 + loss_20 + 2*loss_21+ loss_mean
                
                
                tepoch.set_postfix(ut_loss = loss.item())
                    
                optimizer_2.zero_grad()    
                loss.backward()
                optimizer_2.step()
                tepoch.set_description(f"epoch %2f " % epoches)
                

                       
        print("loss g1 (label=0, sensitive=0):",loss00 )
        print("loss g2 (label=0, sensitive=1):",loss01 )
        print("loss g3 (label=1, sensitive=0):",loss10 )
        print("loss g4 (label=1, sensitive=1):",loss11 )
        print("loss g5 (label=2, sensitive=0):",loss20 )
        print("loss g6 (label=2, sensitive=1):",loss21 )
        print('mean loss:', loss_mean.item())
        wga =test(model, classifier, test_data_loader, print_fairness=False)
        if wga > wg:
            wg = wga
            torch.save(model, 'MultiNLI.pth')
            torch.save(classifier, 'classifier.pth')
              
             
        
        

type_id = 'base'      
train_model(type_id)
print('**********************************')
print('************Evaluation************')

BERT = BertModel.from_pretrained("bert-base-uncased")      
model = BERT_base(BERT, type_id).to(device)                
classifier = nn.Linear(768, 3)  
classifier = classifier.to(device)
classifier = torch.load('classifier.pth')
model = torch.load('MultiNLI.pth')
model = model.to(device)
classifier = classifier.to(device)
model.eval()
classifier.eval()
_  = test(model, classifier, test_data_loader, True)


