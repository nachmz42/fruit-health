from src.ml.utils.utils import load_local_model, preprocess_image, load_labels
from src.config.config import IMAGE_PREDICTION_PATH


def predict(fruit:str) -> None:
    model = load_local_model(fruit)
    img_array = preprocess_image(IMAGE_PREDICTION_PATH)
    predictions = model.predict(img_array)
    class_labels = load_labels(fruit)
    for label, prob in zip(class_labels, predictions[0]):
        print(f"Class: {label.split('__')[-1]}, Probability: {prob}")
