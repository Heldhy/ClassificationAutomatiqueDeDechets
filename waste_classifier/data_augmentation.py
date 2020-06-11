from pathlib import Path

from matplotlib.pyplot import figure, tight_layout, imshow, savefig
from numpy import argmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.mobilenet import preprocess_input

from waste_classifier import batch_size, CLASSES, HEIGHT, WIDTH


def create_new_generator(x_train, y_train):
    datagen = ImageDataGenerator(rotation_range=20,
                                 validation_split=0.2,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.5,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 brightness_range=(0.8, 1.2),
                                 vertical_flip=True,
                                 preprocessing_function=preprocess_input)
    datagen.fit(x_train)
    train_generator = datagen.flow(x_train, y_train, batch_size=batch_size, subset='training')
    val_generator = datagen.flow(x_train, y_train, batch_size=batch_size, subset='validation')
    return train_generator, val_generator


def save_generated_batch(generator, batch_number, path="batch_images"):
    current_batch = 0
    path_to_save_at = Path(path)
    if not path_to_save_at.exists():
        path_to_save_at.mkdir(parents=True)
    for X_batch, y_batch in generator:
        if (current_batch == batch_number):
            figure_plot = figure(figsize=(20, 20))
            for current_batch in range(0, batch_size):
                fig = figure_plot.add_subplot(batch_size // 8, 8, current_batch + 1)
                fig.set_title(CLASSES[argmax(y_batch[current_batch])])
                tight_layout()
                imshow(X_batch[current_batch].reshape(HEIGHT, WIDTH, 3))
            savefig((path_to_save_at / ("batch" + str(batch_number))))
            return
        current_batch += 1
