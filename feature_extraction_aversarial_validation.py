import pandas as pd
import numpy as np
from sklearn.externals import joblib

def file2mat(filename: str) -> np.ndarray:
    mat = genfromtxt(f'{data_dir}{filename}', delimiter=',')
    return mat

if __name__ == '__main__':
    train = pd.read_csv('bbdc_2019_Bewegungsdaten/train.csv')
    test = pd.read_csv('bbdc_2019_Bewegungsdaten/challenge.csv')
    
    train = train[train['Label'] != 'lay'].reset_index(drop=True)
    
    train = train.sample(n=1738)
    
    train['Label'] = 0
    test['Label'] = 1
    
    data_pd = pd.concat((train, test))
    train = data_pd.sample(frac=0.8)
    valid = data_pd.drop(train.index)
    data_pd = pd.concat((train, valid), axis=0)

    features = data_pd['Datafile'].apply(file2mat)
    lens = [f.shape[0] for f in features.values]
    features = features.values

    data = {'features': features}
    data['labels'] = {'train': train['Label'].values,
                      'valid': valid['Label'].values}
    data['bounds'] = {'train': train.shape[0],
                      'valid': test.shape[0]}
    data['lens'] = {'train': np.array(lens[:train.shape[0]]),
                    'valid': np.array(lens[train.shape[0]:])}

    joblib.dump(data, 'data_aversarial_valid.pkl')