import torch
from ..models.lstm_model import LSTMModel
from ..models.tcn_model import TCN


def save_model(model_path=None, model=None):
    torch.save(model.state_dict(), model_path)
    return model_path


def load_model(model_path=None, type='lstm', params=None):
    if type == 'lstm':
        model = LSTMModel(**params)
    if type == 'tcn':
        num_channels = [params['input_size']] + [params['hidden_size'] for i in range(params['num_layers']-1)]
        model = TCN(input_size=params['input_size'],
                    output_size=params['output_size'],
                    num_channels=num_channels,
                    kernel_size=3,
                    dropout=0.2)
    model.load_state_dict(torch.load(model_path))
    return model
