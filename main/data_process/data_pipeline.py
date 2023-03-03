from sklearn.pipeline import Pipeline
from missing_value import FillNaRollMean, DropNa

def get_pipeline():
  columns_fillna = ['Close', 'Volume', 'Open', 'High', 'Low']
  pipeline = Pipeline(
    [
        ('fillna_5_mean', FillNaRollMean(columns_fillna)),
        ('drop_na', DropNa())
    ]
  )
  return pipeline