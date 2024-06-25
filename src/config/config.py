import os

LOCAL_MODEL_PATH = os.getenv('LOCAL_MODEL_PATH','models/')
DATASOURCE_PATH = os.getenv('DATASOURCE_PATH','images/')
DATASET_PATH = os.getenv('DATASET_PATH','datasets/')
SPLIT_RATIO = os.getenv('SPLIT_RATIO',0.8)
EPOCHS = os.getenv('EPOCHS',20)
IMAGE_PREDICTION_PATH = os.getenv('IMAGE_PREDICTION_PATH',"image_prediction/img_pred.jpg")
