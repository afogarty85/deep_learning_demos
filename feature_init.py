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

# batch norm
class FF_NN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(FF_NN, self).__init__()
        # initialize 3 layers
        # first hidden layer
        self.linear_1 = torch.nn.Linear(num_features, num_hidden_1)
        self.linear_1_bn = torch.nn.BatchNorm1d(num_hidden_1)
        # second hidden layer
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_hidden_2)
        self.linear_2_bn = torch.nn.BatchNorm1d(num_hidden_2)
        # output layer
        self.linear_out = torch.nn.Linear(num_hidden_2, num_classes)

    # define how and what order model parameters should be used in forward prop.
    def forward(self, x):
        # run inputs through first layer
        out = self.linear_1(x)
        # apply relu
        out = F.relu(out)
        # apply batchnorm
        out = self.linear_1_bn(out)
        # run inputs through second layer
        out = self.linear_2(out)
        # apply relu
        out = F.relu(out)
        # apply batchnorm
        out = self.linear_2_bn(out)
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




# Xavier weight initialization

# nn.Module tells PyTorch to do backward propagation
class FF_NN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(FF_NN, self).__init__()
        # initialize 3 layers
        # first hidden layer
        self.linear_1 = torch.nn.Linear(num_features, num_hidden_1)
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()
        # second hidden layer
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_hidden_2)
        self.linear_2.weight.detach().normal_(0.0, 0.1)
        self.linear_2.bias.detach().zero_()
        # output layer
        self.linear_out = torch.nn.Linear(num_hidden_2, num_classes)
        self.linear_out.weight.detach().normal_(0.0, 0.1)
        self.linear_out.bias.detach().zero_()

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

# automatic Xavier
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.xavier_uniform_(m.bias)
model.apply(weights_init)

# PyTorch does this
def reset_parameters(self):
    stdv = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)
    if self.bias is not None:
        self.bias.data.uniform_(-stdv, stdv)
