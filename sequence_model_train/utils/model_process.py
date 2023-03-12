import torch
import pickle
from ..models.lstm_model import LSTMModel
from ..models.tcn_model import TCN
from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def save_nn_model(model_path=None, model=None):
    torch.save(model.state_dict(), model_path)
    return model_path


def save_model(model_path=None, model=None):
    model.save(model_path)


def save_model_2(model_path=None, model=None):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def load_nn_model(model_path=None, type='lstm', params=None):
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


def load_model(model_path=None):
    model = ARIMAResults.load(model_path)
    return model


def load_model_2(model_path=None):
    model = pickle.load(open(model_path, 'rb'))
    return model


def caculate_eval(pred_y, y):
    mse = mean_squared_error(y, pred_y)
    mae = mean_absolute_error(y, pred_y)
    mape = mean_absolute_percentage_error(y, pred_y)
    return mse, mae, mape
