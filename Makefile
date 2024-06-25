include .env

split_data:
		python -c 'from src.ml.utils.utils import split_all_fruits; split_all_fruits()'

train_all_fruits:
		python -c 'from src.ml.train.train import train_all; train_all(${FRUITS})'
