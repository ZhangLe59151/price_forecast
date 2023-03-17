from sklearn.pipeline import Pipeline
from .missing_value import FillNaRollMean, DropNa, FillNaLastRow


def get_pipeline():
    # columns_fillna = ['Close', 'Volume', 'Open', 'High', 'Low']
    pipeline = Pipeline(
      [
        # ('fillna_5_mean', FillNaRollMean(columns_fillna)),
        ('fillna_last_row', FillNaLastRow()),
        ('drop_na', DropNa())
      ]
    )
    return pipeline
