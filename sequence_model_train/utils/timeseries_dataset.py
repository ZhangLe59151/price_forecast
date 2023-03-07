import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self,
                 data_path,
                 input_steps,
                 output_steps,
                 start_date=None,
                 end_date=None,
                 normalize=False,
                 mean=None,
                 std=None):
        self.data = pd.read_csv(data_path)
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.start_date = start_date
        self.end_date = end_date
        self.normalize = normalize
        self.mean = mean
        self.std = std

        # Filter data by date range
        if start_date is not None and end_date is not None:
            self.data = self.data[(self.data['DATE'] >= start_date) & (self.data['DATE'] <= end_date)]

        # Compute max index for input/output sequences
        self.max_input_index = len(self.data) - input_steps - output_steps

        # drop the date column
        self.data = self.data.drop(columns='DATE')

        # apply normalization
        if self.normalize:
            self.mean = self.data.iloc[:, :-1].mean(axis=0)
            self.std = self.data.iloc[:, :-1].std(axis=0)
        data_label = self.data.iloc[:, -1]
        column_label = self.data.columns.values[-1]
        self.data = (self.data.iloc[:, :-1] - self.mean) / self.std
        self.data[column_label] = data_label

    def get_mean_std(self):
        return self.mean, self.std

    def __len__(self):
        return self.max_input_index

    def num_features(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        # Get input and output sequences
        input_seq = self.data.iloc[idx:idx+self.input_steps, :].values.astype('float32')
        output_seq = self.data.iloc[idx+self.input_steps:idx+self.input_steps+self.output_steps, -1:].values.astype('float32')

        # Convert sequences to PyTorch tensors
        input_seq = torch.from_numpy(input_seq)  # Add batch dimension
        output_seq = torch.from_numpy(output_seq)  # Add batch dimension

        return input_seq, output_seq
