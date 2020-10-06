# load packages
import sys
sys.path.append("C:/Users/Andrew/Desktop/Projects/Deep Learning/utils")  # this is the folder with py files
from tools import AverageMeter, ProgressBar #scriptName without .py extension; import each class
from radam import RAdam
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import time, datetime, random, re, os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms


SEED = 15
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.amp.autocast(enabled=True)

# set gpu/cpu
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# create Dataset
class CSVDataset(Dataset):
    """MNIST dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # initialize
        self.data_frame = pd.read_csv(csv_file)
        # all columns but the last
        self.features = self.data_frame[self.data_frame.columns[:-1]]
        # the last column
        self.target = self.data_frame[self.data_frame.columns[-1]]
        # initialize the transform if specified
        self.transform = transform

        # get length of df
    def __len__(self):
        return len(self.data_frame)

        # get sample target
    def __get_target__(self):
        return self.target

        # get df filtered by indices
    def __get_values__(self, indices):
        return self.data_frame.iloc[indices]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # pull a sample in a dict
        sample = {'features': torch.tensor(self.features.iloc[idx].values),
                  'target': torch.tensor(self.target.iloc[idx]),
                  'idx': torch.tensor(idx)}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Pixel_Normalize():

    # retrieve sample and unpack it
    def __call__(self, sample):
        features, target, idx = (sample['features'],
                              sample['target'],
                              sample['idx'])

        # normalize each pixel
        normalized_pixels = torch.true_divide(sample['features'], 255)

        # yield another dict
        return {'features': normalized_pixels,
                'target': target,
                'idx': idx}


# instantiate the lazy data set
csv_dataset = CSVDataset(csv_file='https://datahub.io/machine-learning/mnist_784/r/mnist_784.csv',
                         transform=Pixel_Normalize())

# set train, valid, and test size
train_size = int(0.8 * len(csv_dataset))
valid_size = int(0.1 * len(csv_dataset))

# use random split to create three data sets;
train_ds, valid_ds, test_ds = torch.utils.data.random_split(csv_dataset, [train_size, valid_size, valid_size])

# check the output
for i, batch in enumerate(train_ds):
    if i == 0:
        break


# check the distribution of dependent variable; some imbalance
csv_dataset.__get_target__().value_counts()


# prepare weighted sampling for imbalanced classification
def create_sampler(train_ds, csv_dataset):
    # get indicies from train split
    train_indices = train_ds.indices
    # generate class distributions [y1, y2, etc...]
    bin_count = np.bincount(csv_dataset.__get_target__()[train_indices])
    # weight gen
    weight = 1. / bin_count.astype(np.float32)
    # produce weights for each observation in the data set
    samples_weight = torch.tensor([weight[t] for t in csv_dataset.__get_target__()[train_indices]])
    # prepare sampler
    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight,
                                                     num_samples=len(samples_weight),
                                                     replacement=True)
    return sampler


# create sampler for the training ds
train_sampler = create_sampler(train_ds, csv_dataset)

# create NN
# nn.Module tells PyTorch to do backward propagation
class FF_NN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(FF_NN, self).__init__()
        # initialize 3 layers
        # first hidden layer
        self.linear_1 = torch.nn.Linear(num_features, num_hidden_1)
        # second hidden layer
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_hidden_2)
        # output layer
        self.linear_out = torch.nn.Linear(num_hidden_2, num_classes)

    # define how and what order model parameters should be used in forward prop.
    def forward(self, x):
        # run inputs through first layer
        out = self.linear_1(x)
        # apply relu
        out = F.relu(out)
        # run inputs through second layer
        out = self.linear_2(out)
        # apply relu
        out = F.relu(out)
        # run inputs through final classification layer
        logits = self.linear_out(out)
        probs = F.log_softmax(logits, dim=1)
        return logits, probs


# load the NN model
num_features = 784
num_hidden_1 = 128
num_hidden_2 = 256
num_classes = 10
model = FF_NN(num_features=num_features, num_classes=num_classes).to(DEVICE)

# optimizer
optimizer = RAdam(model.parameters(), lr=0.1)

# set number of epochs
epochs = 4


# create DataLoaders with samplers
train_dataloader = DataLoader(train_ds,
                              batch_size=100,
                              sampler=train_sampler,
                              shuffle=False)

valid_dataloader = DataLoader(valid_ds,
                              batch_size=100,
                              shuffle=True)

test_dataloader = DataLoader(test_ds,
                              batch_size=100,
                              shuffle=True)

# set LR scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=0.01,
                                                total_steps=len(train_dataloader)*epochs)

# create gradient scaler for mixed precision
scaler = GradScaler()

# train function
def train(dataloader):
    pbar = ProgressBar(n_total=len(dataloader), desc='Training')
    train_loss = AverageMeter()
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        b_features, b_target, b_idx = batch['features'].to(DEVICE),  batch['target'].to(DEVICE), batch['idx'].to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            logits, probs = model(b_features)
            loss = F.cross_entropy(logits, b_target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        pbar(step=batch_idx, info={'loss': loss.item()})
        train_loss.update(loss.item(), n=1)
    return {'loss': train_loss.avg}


# valid/test function
def test(dataloader):
    pbar = ProgressBar(n_total=len(dataloader), desc='Testing')
    valid_loss = AverageMeter()
    valid_acc = AverageMeter()
    valid_f1 = AverageMeter()
    model.eval()
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            b_features, b_target, b_idx = batch['features'].to(DEVICE),  batch['target'].to(DEVICE), batch['idx'].to(DEVICE)
            logits, probs = model(b_features)
            loss = F.cross_entropy(logits, b_target).item()
            pred = probs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(b_target.view_as(pred)).sum().item()
            f1 = f1_score(pred.to("cpu").numpy(), b_target.to("cpu").numpy(), average='macro')
            valid_f1.update(f1, n=b_features.size(0))
            valid_loss.update(loss, n=b_features.size(0))
            valid_acc.update(correct, n=1)
            count += b_features.size(0)
            pbar(step=batch_idx)
    return {'valid_loss': valid_loss.avg,
            'valid_acc': valid_acc.sum /count,
            'valid_f1': valid_f1.avg}

# training
for epoch in range(1, epochs + 1):
    train_log = train(train_dataloader)
    valid_log = test(valid_dataloader)
    logs = dict(train_log, **valid_log)
    show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
    print(show_info)

# testing
test_log = test(test_dataloader)
print(test_log)



#
