from colorama import Fore, Style
import glob
import os
from PIL import Image
import time
from tensorflow.keras.models import load_model
import shutil
import numpy as np
from src.config.config import LOCAL_MODEL_PATH, DATASOURCE_PATH, DATASET_PATH
import json

def save_model(model, fruit: str) -> None:
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # e.g. 20210824-154952

    # Save Model locally
    model_path =  LOCAL_MODEL_PATH + f"{fruit}_{timestamp}.h5"
    model.save(model_path)

    print(f"✅ {fruit} Model saved locally")

    return None

def load_local_model(fruit: str):
    #load model locally
    local_model_paths = glob.glob(os.path.join(LOCAL_MODEL_PATH, f'{fruit}_*'))
    if not local_model_paths:
        print(Fore.YELLOW +
                f"⚠️ No {fruit} model found in {LOCAL_MODEL_PATH}"
                + Style.RESET_ALL)
        raise FileNotFoundError

    most_recent_model_path_on_disk = sorted(
        local_model_paths)[-1]

    print(f"✅ {fruit} Model found at {most_recent_model_path_on_disk}")
    print(Fore.BLUE + f"\nLoad latest {fruit} model from disk..." + Style.RESET_ALL)

    latest_model = load_model(most_recent_model_path_on_disk)

    print(f"✅ {fruit} Model loaded from local disk")

    return latest_model

def split_all_fruits():
    fruits_paths = os.listdir(DATASOURCE_PATH)
    for fruit_path in fruits_paths:
        fruit = fruit_path.split('__')[0]
        new_fruit_path = f'{DATASET_PATH}/{fruit}/'
        if not os.path.exists(new_fruit_path):
            os.makedirs(new_fruit_path)

        shutil.move(f'{DATASOURCE_PATH}/{fruit_path}', new_fruit_path)


def preprocess_image(image_path, target_size=(224, 224)):

    image = Image.open(image_path)
    image = image.resize(target_size)

    image_array = np.array(image)

    if image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def load_labels(fruit:str) -> str:
    with open(f'{fruit}_class_labels.json', 'r') as f:
        class_labels = json.load(f)
    return class_labels
