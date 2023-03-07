import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# load data
def load_dataset(data_path=None, dtype='pandas'):
    data = pd.read_csv(data_path)
    return data


# Split the dataset into training and validation sets
def split_train_valid(data):
    '''
    Split the dataset into training and validation sets
    where 80% of the data is used for training
    and 20% is used for validation.
    '''
    train_size = int(data.shape[0] * 0.8)
    train_dataset, valid_dataset = data[0:train_size], data[train_size:]
    return train_dataset, valid_dataset


# Normalization
def get_normalization(data_fit,
                      data_transform=None,
                      action_tranform=False):
    scaler = StandardScaler()
    data_x = scaler.fit_transform(data_fit[:, :-1])
    data_y = data_fit[:, -1]
    if action_tranform:
        data_x = scaler.transform(data_transform[:, :-1])
        data_y = data_transform[:, -1]
    data = np.concatenate((data_x, data_y.reshape(-1, 1)), axis=1)
    return data
