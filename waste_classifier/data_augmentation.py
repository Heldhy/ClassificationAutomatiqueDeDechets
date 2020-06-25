from pathlib import Path

from matplotlib.pyplot import figure, tight_layout, imshow, savefig
from numpy import argmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.mobilenet import preprocess_input

from waste_classifier import batch_size, CLASSES, HEIGHT, WIDTH


def create_new_generator(x_train, y_train):
    """
    The parameters of the ImageDataGenerator have been chosen empirically to give the best accuracy and recall after
    training the model
    :return: a tuple of iterators to pass to the model for training
    """
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
    path_to_save_at = Path(path)
    if not path_to_save_at.exists():
        path_to_save_at.mkdir(parents=True)
    for current_batch, (X_batch, y_batch) in enumerate(generator):
        if current_batch == batch_number:
            figure_plot = figure(figsize=(20, 20))
            for current_image in range(batch_size):
                fig = figure_plot.add_subplot(batch_size // 8, 8, current_image + 1)
                fig.set_title(CLASSES[argmax(y_batch[current_image])])
                tight_layout()
                imshow(X_batch[current_image].reshape(HEIGHT, WIDTH, 3))
            savefig((path_to_save_at / ("batch" + str(batch_number))))
            return
