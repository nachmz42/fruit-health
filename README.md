# Fruit Health Detection with ResNet50 🍓

![Fruit Health](https://img.shields.io/badge/Fruit_Health-Detection-green)

This project utilizes a ResNet50 model from Keras to classify images of fruits into healthy and rotten categories. It leverages the dataset from [Fruit and Vegetable Disease - Healthy vs Rotten](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten).

## Installation ⏳

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/nachmz42/fruit-health.git
cd fruit-health
pip install -r requirements.txt
```

## Usage 🐱‍👤

### Setting Up

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten) and place it in the `datasets/` directory.

2. Set up your environment variables in a `.env` file:

   ```dotenv
   LOCAL_MODEL_PATH = "models/"
   FRUITS = ["Jujube", "Mango", "Cucumber", "Pomegranate", "Orange", "Guava", "Strawberry", "Tomato", "Banana", "Apple", "Potato", "Grape", "Carrot", "Bellpepper"]
   EPOCHS = 1
   DATASOURCE_PATH = "images/"
   DATASET_PATH = "datasets/"
   SPLIT_RATIO = 0.8
   IMAGE_PREDICTION_PATH = "image_prediction/img_pred.jpg"
   ```

### Training

To split the data and train the model for all specified fruits, run the following commands:

```bash
make split_data
make train_all_fruits
```

make split_data: This command will execute the split_all_fruits function from src.ml.utils.utils to split the dataset into new paths different from the originals in the Kaggle site in order to facilitate the usage of the trianing function.

make train_all_fruits: This command will execute the train_all function from src.ml.train.train, training the ResNet50 model on all the fruits listed in the FRUITS variable in the .env file, for the number of epochs specified by EPOCHS.

### Prediction

Once the model is trained, you can use it to predict the health status of a fruit image. Follow the steps below to make predictions:

1. **Prepare the Image**: Ensure you have the image you want to predict stored in the path specified by `IMAGE_PREDICTION_PATH` in the `.env` file.

2. **Run the Prediction Script**:

   from src.ml.predict.predict import predict;
   predict("Apple")

You need to specify the fruit you're trying to predict and use the function on the predict script.
