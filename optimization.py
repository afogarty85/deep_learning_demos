# import packages
import torch
from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append("C:/Users/Andrew/Desktop/Projects/Deep Learning/utils")
from tools import AverageMeter, ProgressBar


SEED = 15
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.amp.autocast(enabled=True)

# set torch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# create Dataset
class CSVDataset(Dataset):
    """LM dataset."""

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

        # get df mean and std
    def __get_norm__(self):
        self.mu, self.sigma = np.mean(self.features.values, axis=0), np.std(self.features.values, axis=0)
        return self.mu, self.sigma

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


class Standardize():
    # retrieve sample and unpack it
    def __call__(self, sample):
        features, target, idx = (sample['features'],
                              sample['target'],
                              sample['idx'])

        # normalize each value
        normalized_features = (features - csv_dataset.__get_norm__()[0]) / csv_dataset.__get_norm__()[1]

        # yield another dict
        return {'features': torch.as_tensor(normalized_features,
                                         dtype=torch.float32,
                                         device=device),
                'target': torch.as_tensor(target,
                                          dtype=torch.float32,
                                          device=device),
                'idx': torch.as_tensor(idx,
                                       dtype=torch.int,
                                       device=device)}

X, y = datasets.make_classification(n_samples=1000,
                                         n_features=2,
                                         n_informative=2,
                                         n_redundant=0,
                                         n_classes=2,
                                         random_state=15)

df = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'y': y})
df.to_csv('classification_demo.csv', index=False)

# instantiate the lazy data set
csv_dataset = CSVDataset(csv_file='classification_demo.csv', transform=Standardize())

# check normalization unit variance values
csv_dataset.__get_norm__()[1]
csv_dataset.__get_norm__()[0]

# check some data
for i, batch in enumerate(csv_dataset):
    if i == 0:
        break

# set train and test size
train_size = int(0.8 * len(csv_dataset))
test_size = int(0.2 * len(csv_dataset))

# split data sets
train_ds, test_ds = torch.utils.data.random_split(csv_dataset, [train_size, test_size])

# create manual logistic regression model
class LR():
    def __init__(self, num_features, LAMBDA=0.0):
        self.num_features = num_features
        self.weights = torch.zeros(1, num_features,
                                   dtype=torch.float32, device=device)
        self.bias = torch.zeros(1, dtype=torch.float32, device=device)
        self.LAMBDA = LAMBDA

    def forward(self, x):
        linear = torch.add(torch.mm(x, self.weights.t()), self.bias).view(-1)
        probas = torch.sigmoid(linear)
        return probas

    def backward(self, x, y, probas):
        grad_loss_out = y - probas.view(-1)
        grad_loss_w = -torch.mm(x.t(), grad_loss_out.view(-1, 1)).t()
        grad_loss_b = -torch.sum(grad_loss_out)
        return grad_loss_w, grad_loss_b

    def predict_labels(self, x):
        probas = self.forward(x)
        labels = torch.where(probas > 0.5, torch.tensor(1, device=device), torch.tensor(0, device=device)).view(-1)
        return labels

    def evaluate(self, x, y):
        labels = self.predict_labels(x).float()
        accuracy = torch.sum(labels.view(-1) == y.float()).item() / y.size(0)
        return accuracy

    def _logit_cost(self, y, probas):
        tmp1 = torch.mm(-y.view(1, -1), torch.log(probas.view(-1, 1)))
        tmp2 = torch.mm((1 - y).view(1, -1), torch.log(1 - probas.view(-1, 1)))
        l2 = self.LAMBDA / 2.0 * torch.sum(self.weights**2)
        return (tmp1 - tmp2) + l2


# create DataLoaders
train_dataloader = DataLoader(train_ds,
                              batch_size=100,
                              sampler=None,
                              shuffle=True)

test_dataloader = DataLoader(test_ds,
                              batch_size=100,
                              sampler=None,
                              shuffle=False)

# check data
for i, batch in enumerate(train_dataloader):
    if i == 0:
        break

# set epochs
epochs = 20

# set lr
learning_rate = 0.01

# instantiate model
model = LR(num_features=2, LAMBDA=25.0)


# prepare training function
def train(dataloader):
    pbar = ProgressBar(n_total=len(dataloader), desc='Training')
    train_loss = AverageMeter()
    for batch_idx, batch in enumerate(dataloader):
        # forward
        probas = model.forward(batch['features'])
        # backward
        grad_w, grad_b = model.backward(batch['features'],
                                        batch['target'],
                                        probas)
        # manual regularization -- account for mini-batches
        l2_reg = model.LAMBDA * model.weights / len(dataloader)
        # update weights
        model.weights -= learning_rate * (grad_w + l2_reg)
        model.bias -= learning_rate * grad_b
        # record loss
        loss = model._logit_cost(batch['target'], probas)
        # update meter
        train_loss.update(loss.item(), n=1)
        # update progress bar
        pbar(step=batch_idx, info={'batch_loss': loss.item()})
    return {'train_loss': train_loss.avg}


# training
for epoch in range(1, epochs + 1):
    train_log = train(train_dataloader)
    logs = dict(train_log)
    train_logs = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
    print(train_logs)

print('Weights', model.weights)
print('Bias', model.bias)


# valid/test function
def test(dataloader):
    pbar = ProgressBar(n_total=len(dataloader), desc='Testing')
    valid_loss = AverageMeter()
    valid_acc = AverageMeter()
    count = 0
    for batch_idx, batch in enumerate(dataloader):
        # forward -- skip backward prop
        probas = model.forward(batch['features'])
        # record loss
        loss = model._logit_cost(batch['target'], probas)
        # get predictions
        prediction = torch.where(probas > 0.5, torch.tensor(1, device=device), torch.tensor(0, device=device)).view(-1)
        # compare
        correct = prediction.eq(batch['target']).sum().item()
        valid_loss.update(loss, n=batch['features'].size(0))
        valid_acc.update(correct, n=1)
        count += batch['features'].size(0)
        pbar(step=batch_idx)
    return {'valid_loss': valid_loss.avg,
            'valid_acc': valid_acc.sum / count}

# testing
test_log = test(test_dataloader)
print(test_log)



# standard logistic
X, y = datasets.make_classification(n_samples=1000,
                                         n_features=2,
                                         n_informative=2,
                                         n_redundant=0,
                                         n_classes=2,
                                         random_state=15)

# Normalize (mean zero, unit variance)
mu, sigma = np.mean(X, axis=0), np.std(X, axis=0)
X = (X - mu) / sigma

clf = LogisticRegression(solver='lbfgs', penalty='none').fit(X, y)
clf.coef_
clf.intercept_

## sklearn verify
# C = inverse of lambda
LAMBDA = 25.0
C = 1 / LAMBDA
clf = LogisticRegression(solver='lbfgs', penalty='l2', C=C).fit(X, y)
clf.coef_
clf.intercept_
