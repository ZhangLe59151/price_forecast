import torch
from ..models.lstm_model import LSTMModel


def save_model(model_path=None, model=None):
    torch.save(model.state_dict(), model_path)
    return model_path


def load_model(model_path=None, type='lstm', params=None):
    if type == 'lstm':
        model = LSTMModel(**params)
    model.load_state_dict(torch.load(model_path))
    return model
