import pandas as pd
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

    def set_column_order(self, target_label):
        column_order = [col for col in self.df.columns if col != target_label] + [target_label]
        self.df = self.df.reindex(columns=column_order)

    def generate(self):
        self.load_data()
        self.get_date_featuers('DATE')
        self.set_column_order('USEP')
        pipeline = get_pipeline()
        pipeline.fit(self.df)
        data = pipeline.transform(self.df)
        self.data = data
        if self.save_data:
            self.df.to_csv(self.save_path, index=False)
        return self.df
