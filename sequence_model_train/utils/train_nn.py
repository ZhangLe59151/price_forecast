import torch
import torch.nn as nn

from ..models.lstm_model import LSTMModel
from ..models.tcn_model import TCN


class TrainNNModel:
    def __init__(self, params, model_type):
        self.model_type = model_type
        self.model_params = {
            'input_size': params['input_size'],
            'hidden_size': params['hidden_size'],
            'num_layers': params['num_layers'],
            'output_size': params['output_size'],
            'num_epochs': params['num_epochs']
        }
        self.train_dataloader = None
        self.valid_dataloader = None
        self.model = None
        self.result = None

    def fit(self, X, y):
        params = {
            'input_size': self.model_params['input_size'],
            'hidden_size': self.model_params['hidden_size'],
            'num_layers': self.model_params['num_layers'],
            'output_size': self.model_params['output_size'],
        }
        if self.model_type == 'lstm':
            model = LSTMModel(**params)
        if self.model_type == 'tcn':
            num_channels = [params['input_size']] + [params['hidden_size'] for i in range(params['num_layers']-1)]
            model = TCN(input_size=params['input_size'],
                        output_size=params['output_size'],
                        num_channels=num_channels,
                        kernel_size=3,
                        dropout=0.2)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters())
        train_loss = []
        valid_loss = []
        for epoch in range(self.model_params['num_epochs']):
            loss_0 = 0.0
            num_samples = 0.0
            for i, (inputs, labels) in enumerate(X):
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
            for i, (inputs, labels) in enumerate(y):
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
            for inputs, targets in y:
                outputs = model(inputs)
                loss = criterion(outputs, targets.squeeze())
                mae = mae_func(outputs, targets.squeeze())
                mape = mape_func((outputs - targets.squeeze()).abs()
                                 / targets.squeeze(),
                                 torch.zeros_like(targets.squeeze()))
                total_loss += loss.item() * len(inputs)
                total_mae += mae.item() * len(inputs)
                total_mape += mape.item() * len(inputs)
                num_samples += len(inputs)
        self.model = model
        self.result = dict(
            model=self.model_type,
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
        return self.model

    def get_base_line(self):
        return self.result
