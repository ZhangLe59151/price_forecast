import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from ..data.timeseries_dataset import TimeSeriesDataset


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


# prepare dataloader for nn model
def get_dataloader(data_path,
                   n_in,
                   n_out,
                   train_start_date,
                   train_end_date,
                   valid_start_date,
                   valid_end_date,
                   batch_size):
    train_dataset = TimeSeriesDataset(data_path,
                                      n_in,
                                      n_out,
                                      train_start_date,
                                      train_end_date,
                                      True)
    mean_, std_ = train_dataset.get_mean_std()
    valid_dataset = TimeSeriesDataset(data_path,
                                      n_in,
                                      n_out,
                                      valid_start_date,
                                      valid_end_date,
                                      False)
    n_features = train_dataset.num_features()
    # create data loaders
    train_loader = DataLoader(train_dataset,
                              batch_size,
                              shuffle=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size,
                              shuffle=False)
    return train_loader, valid_loader, n_features


def get_uniq_data(data_path, n_out=None):
    df = load_dataset(data_path, dtype='pandas')
    if n_out:
        train_size = df.shape[0] - n_out
        df_train = df[0:train_size]
        df_valid = df[train_size:]
    else:
        df_train, df_valid = split_train_valid(df)
    return df_train, df_valid
