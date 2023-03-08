from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel():
    def __init__(self, output_size):
        self.order = (2, 1, 1)
        self.n_out = output_size
        self.model = None
        self.label_set = None

    def fit(self, X):
        train_size = int(X.shape[0] - self.n_out)
        train_dataset, valid_dataset = X[0:train_size], X[train_size:]
        airma_model = ARIMA(train_dataset.iloc[:, -1:].values,
                            order=self.order)
        airma_fit = airma_model.fit()
        self.model = airma_fit
        self.label_set = valid_dataset.iloc[:, -1:].values

    def predict(self, steps):
        pred_y = self.model.forecast(steps)
        return pred_y

    def get_base_line(self):
        pred_y = self.predict(self.n_out)
        mse = mean_squared_error(self.label_set - pred_y)
        mae = mean_absolute_error(self.label_set, pred_y)
        mape = mean_absolute_percentage_error(self.label_set, pred_y)
        return dict(
            model='arima',
            valid_result=dict(
                valid_loss=mse,
                valid_mae=mae,
                valid_mape=mape
            ),
            train_process=dict()
        )
