# price_forecast
framework for timeseries task

# quick start
```
train = TrainModel('/path/to/data')   
train.update_params(n_in = 60, n_out = 7, batch_size = 128, hidden_size = 128, num_epochs = 100)   
train.train()   
```
