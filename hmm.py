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
from hmmlearn import hmm


def standard_scale(data):
    features = data['features']
    lens = np.concatenate((data['lens']['train'],data['lens']['valid'],data['lens']['test']))
    features = np.concatenate(features, axis=0)
    features = StandardScaler().fit_transform(features)
    ret = []
    l = 0
    for t in lens:
        ret.append(features[l: l + t])
        l += t
    features = np.array(ret)
    data['features'] = features


def filter_emg(x):
    high = 20/(1000/2)
    low = 450/(1000/2)
    b, a = signal.butter(4, [high, low], btype='bandpass')
    emg_filtered = signal.filtfilt(b, a, x, axis=0)
    return emg_filtered


def get_split(data, split):
    if split == 'train':
        return data['features'][: data['bounds']['train']]
    elif split == 'valid':
        return data['features'][data['bounds']['train']: data['bounds']['train'] + data['bounds']['valid']]
    elif split == 'test':
        return data['features'][data['bounds']['train'] + data['bounds']['valid']:]


def generate_batch_idx(n, batch_size, randomise=False):
    idx = np.arange(0, n)
    if randomise:
        np.random.shuffle(idx)
    for batch_idx in np.arange(0, n, batch_size):
        yield idx[batch_idx:batch_idx+batch_size]


def generate_batches(data, split, batch_size,
                     time_steps=10, stride=5, randomise=False):
    features = get_split(data, split)

    try:
        labels = data['label_encoder'].transform(data['labels'][split])
    except:
        labels = np.zeros(features.shape[0])
    
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
            preds.append(model(b_data, b_lens))
            labels.append(b_labels)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
    return preds, labels


def get_accuracy(preds, labels, le):
    preds = preds.max(dim=1)[1].numpy()
    preds = [le.classes_[i] for i in preds]
    labels = labels.numpy()
    labels = [le.classes_[i] for i in labels]
    return accuracy_score(labels, preds)


def get_dataset(split, window, overlap):
    features = []
    labels = []
    lens = []
    for batch_data, batch_label, batch_lens in generate_batches(data, split, 1024,
                                                                400, 200, False):
        features.extend(batch_data)
        labels.extend(batch_label)
        lens.extend(batch_lens)
    
    features = np.array(features)
    labels = np.array(labels)
    lens = np.array(lens)
    return features, labels, lens


if __name__ == '__main__':
    
    data = joblib.load('data.pkl')
    train_data, train_labels, train_lens = get_dataset('train', 10, 5)

    models = []
    scores = []
    for i in range(22):
        model = hmm.GaussianHMM(n_components=8, covariance_type='diag', n_iter=10000)
        label = i
        idx = (train_labels == label)
        model.fit(np.concatenate(train_data[idx], axis=0), train_lens[idx])
        score_list = np.array([model.score(seq) for seq in train_data[idx]])
        scores.append({'avg_score': score_list.mean(),
                       'std_score': score_list.std()
                      })
        models.append(model)

    valid_data, valid_labels, valid_lens = get_dataset('valid', 10, 5)

    valid_preds = []
    for seq in valid_data:
        val_scores = [model.score(seq) for model in models]
        pred = np.argmax(val_scores)
        valid_preds.append(pred)

    print(accuracy_score(valid_labels, valid_preds))

    test_data, test_labels, test_lens = get_dataset('test', 10, 5)

    test_preds = []
    for seq in test_data:
        test_scores = [model.score(seq) for model in models]
        pred = np.argmax(test_scores)
        test_preds.append(pred)

    test_preds = [data['label_encoder'].classes_[x] for x in test_preds]
    challenge = pd.read_csv('bbdc_2019_Bewegungsdaten/challenge.csv')
    challenge['Label'] = test_preds
    challenge.to_csv('submission1.csv', index=False)

