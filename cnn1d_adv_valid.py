import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.externals import joblib
from collections import Counter
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def filter_emg(x):
    high = 20/(1000/2)
    low = 450/(1000/2)
    b, a = signal.butter(4, [high, low], btype='bandpass')
    emg_filtered = signal.filtfilt(b, a, x, axis=0)
    return emg_filtered

def get_split(data, split):
    if split == 'train':
        return data['features']

def generate_batch_idx(n, batch_size, randomise=False):
    idx = np.arange(0, n)
    if randomise:
        np.random.shuffle(idx)
    for batch_idx in np.arange(0, n, batch_size):
        yield idx[batch_idx:batch_idx+batch_size]

def generate_batches(data, split, batch_size,
                     time_steps=10, stride=5, randomise=False):
    features = get_split(data, split)

    labels = data['labels'][split]
    
    lens = data['lens'][split]
    new_features = []
    new_labels = []
    new_lens = []
    for i in range(len(lens)):
        mat = features[i]
        label = labels[i]
        l = lens[i]
        acc_emg = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11]
        mat[:, acc_emg] -= mat[:, acc_emg].mean(axis=0)
        mat[:, :4] = filter_emg(mat[:, :4])
        extracted_steps = []
        for j in range(0, len(mat) - time_steps, stride):
            window = mat[j: j + time_steps, :]
            means = window[:, 4:].mean(axis=0).reshape(1, -1)
            rms = np.sqrt((window[:, :4]**2).mean(axis=0)).reshape(1, -1)
            feature_vector = np.concatenate((rms, means), axis=1).reshape(1, -1)
            
            extracted_steps.append(feature_vector)
        extracted_steps = np.concatenate(extracted_steps, axis=0)
        new_features.append(extracted_steps)
        new_labels.append(label)
        new_lens.append(len(extracted_steps))
    features = np.array(new_features)
    labels = np.array(new_labels)
    lens = np.array(new_lens)
    
    n = len(features)
    for batch_idx in generate_batch_idx(n, batch_size, randomise):
        batch_data = features[batch_idx]
        batch_labels = labels[batch_idx]
        batch_lens = lens[batch_idx]
        yield batch_data, batch_labels, batch_lens


def pad_batch(batch, lens):
    max_len = max(lens)
    batch_size = batch.shape[0]
    num_feature = batch[0].shape[1]
    padded_seqs = np.zeros((batch_size, max_len, num_feature))
    
    for i, l in enumerate(lens):
        padded_seqs[i, :l, :] = batch[i][:l]

    return padded_seqs


def torch_batch(batch, targets):
    return torch.from_numpy(batch).float(), torch.from_numpy(targets).long()

def get_preds(model, data, split, batch_size, time_steps, stride):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for b_data, b_labels, b_lens in generate_batches(data, split, batch_size, 
                                                         time_steps, stride, False):
            b_data = pad_batch(b_data, b_lens)
            b_data, b_labels = torch_batch(b_data, b_labels)
            preds.append(torch.sigmoid(model(b_data, b_lens)))
            labels.append(b_labels)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
    return preds, labels

def get_accuracy(preds, labels):
    preds = (torch.sigmoid(preds).numpy().flatten() >= 0.5).astype(np.float32)
    labels = labels.numpy().flatten()
    return (preds == labels).astype(np.float32).mean()


class HARNet(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=19, out_channels=32,
                               kernel_size=8)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64,
                               kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128,
                               kernel_size=4)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=64,
                               kernel_size=4)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=32,
                               kernel_size=4)

        self.norm1 = nn.BatchNorm1d(32)
        self.norm2 = nn.BatchNorm1d(64)
        self.norm3 = nn.BatchNorm1d(128)
        self.norm4 = nn.BatchNorm1d(64)
        self.norm5 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(.2)
        self.lin1 = nn.Linear(32, 22)
        
    def forward(self, data, lens):
        x = self.conv1(data.transpose(1, 2))
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.norm5(x)
        x = self.lin1(x.mean(dim=2))
        return x

if __name__ == '__main__':
    
    data = joblib.load('data_aversarial_train.pkl')
    
    num_epochs = 10000
    batch_size = 32
    time_steps = 80
    stride = 40
    objective = nn.BCEWithLogitsLoss()
    model = HARNet()
    optimiser = torch.optim.Adam(model.parameters(), weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,
                                                           factor=0.5,
                                                           patience=2,
                                                           verbose=True)
    running_loss = 0
    running_batch = 0
    min_valid_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        with tqdm(enumerate(generate_batches(data, 'train', batch_size,
                                             time_steps, stride, True), 1)) as pbar:
            model.train()
            for batch_num, (batch_data, batch_labels, batch_lens) in pbar:
                batch_data = pad_batch(batch_data, batch_lens)
                batch_data, batch_labels = torch_batch(batch_data, batch_labels)
                optimiser.zero_grad()
                preds = model(batch_data, batch_lens)
                loss = objective(preds, batch_labels.float().view(-1, 1))
                loss.backward()
                optimiser.step()
                running_loss += loss.item()
                running_batch += 1
                pbar.set_description(f'[Epoch: {epoch}] | Batch {batch_num} | Loss: {running_loss/running_batch}')

        valid_preds, valid_labels = get_preds(model, data, 'valid',
                                              64, time_steps, stride)

        valid_loss = objective(valid_preds, valid_labels.float().view(-1, 1)).item()
        scheduler.step(valid_loss)
        if valid_loss < min_valid_loss:
            print(f'Validation loss improved from {min_valid_loss} to {valid_loss}')
            acc = get_accuracy(valid_preds, valid_labels)
            print(f'Validation accuracy: {acc}')
            min_valid_loss = valid_loss
            with open('best_cnn_adv_valid_model.pt', 'wb') as f:
                torch.save(model.state_dict(), f)
        else:
            print('Validation loss did not improve')


    model = HARNet()
    model.load_state_dict(torch.load('best_cnn_adv_valid_model.pt'))

    num_epochs = 10000
    batch_size = 32
    time_steps = 100
    stride = 100
    train_preds, train_labels = get_preds(model, data, 'train',
                                              64, time_steps, stride)


    train_loss = objective(train_preds, train_labels).item()
    acc = get_accuracy(train_preds, train_labels)
    print(acc)
    train_preds = (train_preds.numpy().flatten() >= 0.5).astype(np.float32)

    train = pd.read_csv('bbdc_2019_Bewegungsdaten/train.csv')
    train = train[train['Label'] !='lay'].reset_index(drop=True)

    valid = train.loc[train_preds==1, :].copy().reset_index(drop=True)
    valid.to_csv('valid_adversarial.csv', index=False)

    train.loc[train_preds==0,:].to_csv('train_adversarial.csv', index=False)

