### sequence-model-train
Sequence-model-train is a Python library for implementing Long Short-Term Memory (LSTM) models on time-series data. It takes a CSV file as input and allows you to specify various hyperparameters for the model.   
Github: ​https://github.com/ZhangLe59151/price_forecast   
PyPi: ​https://pypi.org/project/sequence-model-train/   
Document: ​https://zhanglenus.gitbook.io/sequence-model-train-library-document/   
Example on the notebook:    

# quick start
install the library first
```
pip install sequence-model-train
```

Then run the trainModel class.   
```
from sequence_model_train.train_model import TrainModel

train = TrainModel('/path/to/data')   
train.update_params(n_in = 60, n_out = 7, batch_size = 128, hidden_size = 128, num_epochs = 100)   
train.train()   
```

# Documents

