import argparse
from sequence_model_train.utils.train_model import TrainModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train time-series model")
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--n-in', type=int, required=False)
    parser.add_argument('--n-out', type=int, required=False)
    args = parser.parse_args()
    train = TrainModel(args.data_path)
    train.update_params(n_in=args.n_in,
                        n_out=args.n_out,
                        batch_size=128,
                        hidden_size=128,
                        num_epochs=100)
    train.train()
