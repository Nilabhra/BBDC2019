import pandas as pd
import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib


def file2mat(filename: str) -> np.ndarray:
    mat = genfromtxt(f'{data_dir}{filename}', delimiter=',')
    return mat


if __name__ == '__main__':
    train = pd.read_csv('train_adversarial.csv')
    valid = pd.read_csv('valid_adversarial.csv')
    test = pd.read_csv('bbdc_2019_Bewegungsdaten/challenge.csv')

    data = pd.concat((train, valid, test))

    features = data['Datafile'].apply(file2mat)
    lens = [f.shape[0] for f in features.values]
    features = features.values

    data = {'features': features}
    data['labels'] = {'train': train['Label'].values,
    data['bounds'] = {'train': train.shape[0],
                      'valid': valid.shape[0],
                      'test': test.shape[0]}
    data['lens'] = {'train': np.array(lens[:train.shape[0]]),
                    'valid': np.array(lens[train.shape[0]: train.shape[0] + valid.shape[0]]),
                    'test': np.array(lens[train.shape[0] + valid.shape[0]:])}

    labels = np.concatenate((data['labels']['train'], data['labels']['valid']))
    le = LabelEncoder()
    le.fit(labels)

    data['label_encoder'] = le

    joblib.dump(data, 'data_adversarial_complete.pkl')

