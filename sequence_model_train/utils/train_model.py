import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.lstm_model import LSTMModel
from ..models.tcn_model import TCN
from ..data.timeseries_dataset import TimeSeriesDataset
from .model_process import load_model


class TrainModel:
    def __init__(self, data_path):
        self.train_data_path = data_path
        self.features_size = None
        self.n_in = 3
        self.n_out = 3
        self.num_epochs = 10
        self.hidden_size = 32
        self.num_layers = 3
        self.batch_size = 32
        self.train_start_date = None
        self.train_end_date = None
        self.valid_start_date = None
        self.valid_end_date = None
        self.model_type = 'lstm'
        self.model = None
        self.model_path = None
        self.result = None

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
        train_start_date, train data start from date, default = '2022-03-03'
        train_end_date, train data end at date, default = '2022-12-31'
        valid_start_date, valid data start from date, default = '2023-01-01'
        valid_start_date, valid data end at date, default = '2023-02-23'
        model_type,  model type,      default = 'lstm'
        '''
        self.train_data_path = kwargs.get('data_path', self.train_data_path)
        self.n_in = kwargs.get('n_in', self.n_in)
        self.n_out = kwargs.get('n_out', self.n_out)
        self.num_epochs = kwargs.get('num_epochs', self.num_epochs)
        self.hidden_size = kwargs.get('hidden_size', self.hidden_size)
        self.num_layers = kwargs.get('num_layers', self.num_layers)
        self.batch_size = kwargs.get('batch_size', self.batch_size)
        self.train_start_date = kwargs.get('train_start_date', self.train_start_date)
        self.train_end_date = kwargs.get('train_end_date', self.train_end_date)
        self.valid_start_date = kwargs.get('valid_start_date', self.valid_start_date)
        self.valid_end_date = kwargs.get('valid_end_date', self.valid_end_date)
        self.model_type = kwargs.get('model_type', self.model_type)
        self.kernal_size = kwargs.get('kernal_size', self.kernal_size)

    def train(self):
        train_dataset = TimeSeriesDataset(self.train_data_path,
                                          self.n_in,
                                          self.n_out,
                                          start_date=self.train_start_date,
                                          end_date=self.train_end_date,
                                          normalize=True)
        mean_, std_ = train_dataset.get_mean_std()
        valid_dataset = TimeSeriesDataset(self.train_data_path,
                                          self.n_in,
                                          self.n_out,
                                          start_date=self.valid_start_date,
                                          end_date=self.valid_end_date,
                                          normalize=False,
                                          mean=mean_,
                                          std=std_)
        # create data loaders
        train_loader = DataLoader(train_dataset,
                                  self.batch_size,
                                  shuffle=True)
        valid_loader = DataLoader(valid_dataset,
                                  self.batch_size,
                                  shuffle=False)
        # Train
        self.features_size = train_dataset.num_features()
        params = {
            'input_size': self.features_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.n_out,
        }
        if self.model_type == 'lstm':
            model = LSTMModel(**params)
        if self.model_type == 'tcn':
            num_channels = [self.features_size] + [self.hidden_size for i in range(self.num_layers-1)]
            model = TCN(input_size=self.features_size,
                        output_size=self.n_out,
                        num_channels=num_channels,
                        kernel_size=self.kernal_size,
                        dropout=0.2)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters())

        train_loss = []
        valid_loss = []
        for epoch in range(self.num_epochs):
            loss_0 = 0.0
            num_samples = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, labels.squeeze())
                loss.backward()
                optimizer.step()
                loss_0 += loss.item() * len(inputs)
                num_samples += len(inputs)
            train_loss.append(loss_0/num_samples)
            loss_1 = 0.0
            num_samples = 0.0
            for i, (inputs, labels) in enumerate(valid_loader):
                model.eval()
                output = model(inputs)
                loss = criterion(output, labels.squeeze())
                loss_1 += loss.item() * len(inputs)
                num_samples += len(inputs)
            valid_loss.append(loss_1/num_samples)
        # evaluate
        model.eval()
        # calculate mae, mape
        mae_func = nn.L1Loss()
        mape_func = nn.L1Loss()
        num_samples = 0.0
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
        self.model = model
        self.train_result = dict(
            model='lstm',
            valid_result=dict(
                valid_loss=total_loss / num_samples,
                valid_mae=total_mae / num_samples,
                valid_mape=total_mape / num_samples
            ),
            train_process=dict(
                train_loss=train_loss,
                valid_loss=valid_loss
            )
        )
        return self.train_result

    def load_model_(self, model_path=None):
        params = {
            'input_size': self.features_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.n_out,
        }
        self.model = load_model(model_path=model_path,
                                type=self.model_type,
                                params=params)
        return self.model

    def predict(self, model_path=None, pred_data=None):
        model = self.load_model_(model_path)
        model.eval()
        # use the loaded model to make predictioins
        if not isinstance(pred_data, torch.Tensor):
            x = torch.from_numpy(pred_data.values).float()
        if x.shape[1] == self.features_size:
            x = x.unsqueeze(0)
        pred_y = model(x)
        return pred_y.detach().numpy()
