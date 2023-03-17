import pandas as pd
import statsmodels.api as sm
from .data_pipeline import get_pipeline


class GenerateTrainData:
    def __init__(self, data_path=None, save_path=None, save_data=True):
        self.data_path = data_path
        self.save_path = save_path
        self.df = None
        self.save_data = save_data

    def load_data(self, data_path=None):
        if data_path:
            self.data_path = data_path
        data = pd.read_csv(self.data_path)
        self.df = data
        return data

    def get_date_featuers(self, date_column):
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        self.df['DAY'] = self.df[date_column].dt.day
        self.df['MONTH'] = self.df[date_column].dt.month
        self.df['YEAR'] = self.df[date_column].dt.year
        self.df['WEEK'] = self.df[date_column].dt.dayofweek

    def get_trend_season(self, index_name, label_name):
        df = pd.read_csv(self.data_path,
                         parse_dates=[index_name],
                         index_col=index_name)
        result = sm.tsa.seasonal_decompose(
            df[label_name],
            model='additive'
        )
        self.df['trend'] = result.trend.values
        self.df['trend'] = self.df['trend'].fillna(0)
        self.df['season'] = result.seasonal.values
        self.df['season'] = self.df['season'].fillna(0)
        self.df['resid'] = result.resid.values
        self.df['resid'] = self.df['resid'].fillna(0)

    def set_column_order(self, target_label):
        column_order = [col for col in self.df.columns if col != target_label] + [target_label]
        self.df = self.df.reindex(columns=column_order)

    def generate(self):
        self.load_data()
        self.get_date_featuers('DATE')
        self.get_trend_season('DATE', 'USEP')
        self.set_column_order('USEP')
        pipeline = get_pipeline()
        pipeline.fit(self.df)
        self.df = pipeline.transform(self.df)
        if self.save_data:
            self.df.to_csv(self.save_path, index=False)
        return self.df
