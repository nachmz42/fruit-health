import os
os.environ["KERAS_BACKEND"] = "jax"
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from src.ml.utils.utils import save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config.config import DATASET_PATH, EPOCHS, IMAGE_PREDICTION_PATH
from tensorflow.keras.metrics import AUC
import json

IMAGE_WIDTH=224
IMAGE_HEIGHT=224
IMAGE_CHANNELS = 3

def train(fruit: str) -> None:
    print(f"Training model for {fruit} ⏳")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        f'{DATASET_PATH}/{fruit}',
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=32,
        subset='training',
        class_mode='categorical'
    )


    validation_generator = train_datagen.flow_from_directory(
        f'{DATASET_PATH}/{fruit}',
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=32,
        subset='validation',
        class_mode='categorical'
    )
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)


    class_indices = train_generator.class_indices
    class_labels = list(class_indices.keys())

    with open(f'{fruit}_class_labels.json', 'w') as f:
        json.dump(class_labels, f)

    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',AUC(name='auc')])

    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = validation_generator.samples // validation_generator.batch_size

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_steps)
    save_model(model,fruit)

    history_dict = history.history

    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    auc_values = history_dict['auc']

    print(f"Final Training Loss: {loss_values[-1]}")
    print(f"Final Validation Loss: {val_loss_values[-1]}")
    print(f"Final Training Accuracy: {acc_values[-1]}")
    print(f"Final Validation Accuracy: {val_acc_values[-1]}")
    print(f"Final AUC: {auc_values[-1]}")
    print(f'{fruit} model trained ✅')


def train_all(fruits: list) -> None:
    for fruit in fruits:
        train(fruit)
