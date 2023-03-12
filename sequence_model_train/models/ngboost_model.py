from ..utils.model_process import caculate_eval
from ngboost import NGBRegressor


class NGBoostModel:
    def __init__(self, columns, n_in, n_out):
        self.columns = columns
        self.n_in = n_in
        self.n_out = n_out
        self.models = []
        self.result = {}
        self.X_columns = []
        self.y_columns = []

    def fit(self, X, y):
        self.models = []
        columns_ = X.columns.difference(self.columns)
        columns_ = X[columns_].select_dtypes(include=['float']).columns.values
        self.columns = X[self.columns].select_dtypes(
            include=['float']
        ).columns.values
        self.X_columns = columns_
        self.y_columns = self.columns
        for i in range(self.n_out):
            ngb = NGBRegressor(
                n_estimators=100,
                learning_rate=0.01,
                verbose=False
            )
            ngb.fit(X[columns_].values, X[self.columns].values[:, i])
            self.models.append(ngb)
        pred_y = []
        for i in range(self.n_out):
            if (i+1-self.n_out == 0):
                pred_y.append(
                    self.models[i].predict(y[columns_].values[-1:, :])[0]
                )
            else:
                pred_y.append(
                    self.models[i].predict(
                        y[columns_].values[i-self.n_out:i+1-self.n_out, :]
                    )[0]
                )
        mse, mae, mape = caculate_eval(pred_y,
                                       y[self.columns].values[0:1, :][0])
        self.result = dict(
            model='ngboost',
            valid_result=dict(
                valid_loss=mse,
                valid_mae=mae,
                valid_mape=mape
            ),
            train_process=dict()
        )
        return self.models

    def get_base_line(self):
        return self.result
