import torch
from .utils import data_preprocess, model_process
from .utils.train_nn import TrainNNModel
from .models.arima_model import ARIMAModel


class Offline:
    def __init__(self):
        self.model = None
        self.model_type = None
        self.support_models = {
            'nn': ['lstm', 'tcn'],
            'arma': ['arima'],
            'tbase': ['xgb']
        }
        self.data_prams = {
            'train_data_path': None,
            'train_start_date': '2021-01-01',
            'train_end_date': '2023-01-01',
            'valid_start_date': '2023-01-01',
            'valid_end_date': '2023-03-01'
        }
        self.model_params = {
            'n_features': 10,
            'n_in': 30,
            'n_out': 7,
            'batch_size': 32,
            'hidden_size': 32,
            'num_layers': 3,
            'input_size': 10,
            'num_epochs': 100,
            'model_path': None
        }
        self.result = None

    def update_params(self, **kwargs):
        self.model = kwargs.get('model', self.model)
        self.model_type = kwargs.get('model_type', self.model_type)
        self.model_params['n_features'] = kwargs.get(
            'n_features',
            self.model_params['n_features']
        )
        self.model_params['n_in'] = kwargs.get(
            'n_in',
            self.model_params['n_in']
        )
        self.model_params['n_out'] = kwargs.get(
            'n_out',
            self.model_params['n_out']
        )
        self.model_params['batch_size'] = kwargs.get(
            'batch_size',
            self.model_params['batch_size']
        )
        self.model_params['hidden_size'] = kwargs.get(
            'hidden_size',
            self.model_params['hidden_size']
        )
        self.model_params['num_layers'] = kwargs.get(
            'num_layers',
            self.model_params['num_layers']
        )
        self.model_params['num_epochs'] = kwargs.get(
            'num_epochs',
            self.model_params['num_epochs']
        )
        self.model_params['model_path'] = kwargs.get(
            'model_path',
            self.model_params['model_path']
        )
        self.data_prams['train_data_path'] = kwargs.get(
            'train_data_path',
            self.data_prams['train_data_path']
        )
        self.data_prams['train_start_date'] = kwargs.get(
            'train_start_date',
            self.data_prams['train_start_date']
        )
        self.data_prams['train_end_date'] = kwargs.get(
            'train_end_date',
            self.data_prams['train_end_date']
        )
        self.data_prams['valid_start_date'] = kwargs.get(
            'valid_start_date',
            self.data_prams['valid_start_date']
        )
        self.data_prams['valid_end_date'] = kwargs.get(
            'valid_end_date',
            self.data_prams['valid_end_date']
        )

    def train_model(self):
        if self.model_type in self.support_models['nn']:
            t_, v_, n_ = data_preprocess.get_dataloader(
                self.data_prams['train_data_path'],
                self.model_params['n_in'],
                self.model_params['n_out'],
                self.data_prams['train_start_date'],
                self.data_prams['train_end_date'],
                self.data_prams['valid_start_date'],
                self.data_prams['valid_end_date'],
                self.model_params['batch_size'])
            self.model_params['n_features'] = n_
            params = {
                'input_size': n_,
                'hidden_size':  self.model_params['hidden_size'],
                'num_layers': self.model_params['num_layers'],
                'output_size': self.model_params['n_out'],
                'num_epochs': self.model_params['num_epochs'],
            }
            model = TrainNNModel(params, self.model_type)
        if self.model_type in self.support_models['arma']:
            t_, v_ = data_preprocess.get_uniq_data(
                self.data_prams['train_data_path'],
                self.model_params['n_out'])
            model = ARIMAModel(self.model_params['n_out'])
        self.model = model.fit(t_, v_)
        self.result = model.get_base_line()
        return self.result

    def save_model(self):
        if self.model_type in self.support_models['nn']:
            model_process.save_nn_model(self.model_params['model_path'],
                                        self.model)
        if self.model_type in self.support_models['arma']:
            model_process.save_model(self.model_params['model_path'],
                                     self.model)

    def load_model(self):
        if self.model_type in self.support_models['nn']:
            self.model = model_process.load_nn_model(
                self.model_params['model_path'],
                self.model_type,
                params={
                    'input_size': self.model_params['n_features'],
                    'hidden_size': self.model_params['hidden_size'],
                    'num_layers': self.model_params['num_layers'],
                    'output_size': self.model_params['n_out'],
                }
            )
        if self.model_type in self.support_models['arma']:
            self.model = model_process.load_model(
                self.model_params['model_path']
            )

    def predict(self, X):
        if self.model_type in self.support_models['nn']:
            self.model.eval()
            if not isinstance(X, torch.Tensor):
                x = torch.from_numpy(X.values).float()
            if x.shape[1] == self.model_params['n_features']:
                x = x.unsqueeze(0)
            pred_y = self.model(x).detach().numpy()
        if self.model_type in self.support_models['arma']:
            pred_y = self.model.forecast(self.model_params['n_out'])
        return pred_y
