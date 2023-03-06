import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from .data_process import data_pipeline
from .lstm_model import LSTMModel

class TrainModel:
    def __init__(self, data_path):
        self.train_data_path = data_path
        self.n_in = 3
        self.n_out = 3
        self.num_epochs = 10
        self.hidden_size = 32
        self.num_layers = 3
        self.batch_size = 32
        self.X = {}
        self.y = {}
    
    # Update parameters
    def update_params(self, **kwargs):
        '''
        data_path,  path/to/traning/data
        n_in,        sequence length, default = 3
        n_out,       output size,     default = 5
        num_epochs,  number of epochs,default = 10
        hidden_size, hidden_size,     default = 32
        num_layers,  number of layers,default = 3
        batch_size,  batch size,      default = 32
        '''
        self.train_data_path = kwargs.get('data_path', self.train_data_path)
        self.n_in = kwargs.get('n_in', self.n_in)
        self.n_out = kwargs.get('n_out', self.n_out)
        self.num_epochs = kwargs.get('num_epochs', self.num_epochs)
        self.hidden_size = kwargs.get('hidden_size', self.hidden_size)
        self.num_layers = kwargs.get('num_layers', self.num_layers)
        self.batch_size = kwargs.get('batch_size', self.batch_size)
    
    # load data
    def load_dataset(self, data_path = None, dtype = 'pandas'):
        data = pd.read_csv(data_path)
        return data
    
    # processinig_data
    def process_data(self, data):
        pipeline = data_pipeline.get_pipeline()
        pipeline.fit(data)
        data = pipeline.transform(data)
        data = data.drop(columns = 'DATE')
        return data

    # generate the sequence
    def create_sequence(self, sequence, n_in, n_out):
        '''
        1.Retrieve all features and historical labels as training input, with N_in steps as the input length. 
        2.Retrieve the next N_out steps label as the output (prediction) for the training data.
        3.The label column in the dataset is located in the last column
        '''
        X, y = list(), list()
        loop_len = sequence.shape[0]
        for i in range(1, loop_len):
            end_idx = i + n_in
            if end_idx+n_out-1 > loop_len:
                break
            seq_x, seq_y = sequence[i-1: end_idx-1, :], sequence[end_idx-1:end_idx+n_out-1, -1:]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X, dtype=float), np.array(y, dtype=float)
    
    # Split the dataset into training and validation sets
    def split_train_valid(self, data):
        '''
        Split the dataset into training and validation sets
        where 80% of the data is used for training 
        and 20% is used for validation.
        '''
        train_size = int(data.shape[0] * 0.8)
        train_dataset, valid_dataset = data[0:train_size], data[train_size:]
        return train_dataset, valid_dataset
    
    # Normalization
    def get_normalization(self, data_fit, data_transform = None, action_tranform = False):
       scaler = StandardScaler()
       data_x = scaler.fit_transform(data_fit[:, :-1])
       data_y = data_fit[:, -1]
       if action_tranform:
          data_x = scaler.transform(data_transform[:, :-1])
          data_y = data_transform[:, -1]
       data = np.concatenate((data_x, data_y.reshape(-1, 1)), axis=1)
       return data

    # prepare the traning data and valid data
    def get_train_valid_dataset(self):
        data = self.load_dataset(self.train_data_path)
        data = self.process_data(data)
        # split dataset
        train_dataset, valid_dataset = self.split_train_valid(data.values)
        # normalization
        train_dataset = self.get_normalization(train_dataset)
        valid_dataset = self.get_normalization(train_dataset, valid_dataset, action_tranform = True)
        # convert to tensor
        train_x, train_y = self.create_sequence(train_dataset, self.n_in, self.n_out)
        valid_x, valid_y = self.create_sequence(valid_dataset, self.n_in, self.n_out)
        (self.X['train'], self.X['valid']) = (torch.from_numpy(train_x).float(), torch.from_numpy(valid_x).float())
        (self.y['train'], self.y['valid']) = (torch.from_numpy(train_y).float(), torch.from_numpy(valid_y).float())

    def train(self):
        self.get_train_valid_dataset()

        # create pytorch datasets
        train_dataset = TensorDataset(self.X['train'], self.y['train'])
        valid_dataset = TensorDataset(self.X['valid'], self.y['valid'])
        # create data loaders
        train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, self.batch_size, shuffle=False)
        
        # Train
        input_size = self.X['train'].shape[2]
        params = {
            'input_size': input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.n_out,
        }
        model = LSTMModel(**params)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters())

        for epoch in range(self.num_epochs):
          for i, (inputs, labels) in enumerate(train_loader):
            output = model(inputs)
            loss = criterion(output, labels.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # evaluate
        model.eval()
        
        # calculate mae, mape
        mae_func = nn.L1Loss()
        mape_func = nn.L1Loss()

        num_samples = 0.01  
        total_loss = 0
        total_mae = 0
        total_mape = 0

        with torch.no_grad():
          for inputs, targets in valid_loader:
            outputs = model(inputs)

            loss = criterion(outputs, targets.squeeze())
            mae = mae_func(outputs, targets.squeeze())
            mape = mape_func((outputs - targets.squeeze()).abs() / targets.squeeze(), torch.zeros_like(targets.squeeze()))

            total_loss += loss.item() * len(inputs)
            total_mae += mae.item() * len(inputs)
            total_mape += mape.item() * len(inputs)
            num_samples += len(inputs)

        return dict(
            model = 'lstm',
            valid_result = dict(
                valid_loss = total_loss / num_samples,
                valid_mae  = total_mae / num_samples,
                valid_mape = total_mape / num_samples
            )
        )