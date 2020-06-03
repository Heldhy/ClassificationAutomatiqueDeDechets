from pathlib import Path

from matplotlib.pyplot import figure, tight_layout, imshow, savefig
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from waste_classifier import batch_size, CLASSES, HEIGHT, WIDTH, model_type, return_preprocessing_function


def create_only_one_generator_for_feature_extraction(x_train, y_train):
    datagen = ImageDataGenerator(rotation_range=20,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.5,
                                 zoom_range=(0.9, 1.1),
                                 preprocessing_function=return_preprocessing_function("mobilenet"))
    datagen.fit(x_train)
    data_generator = datagen.flow(x_train, y_train, batch_size=batch_size)
    return data_generator


def create_new_generator(x_train, y_train, type_of_model=None):
    if (type_of_model is None):
        type_of_model = model_type
    datagen = ImageDataGenerator(rotation_range=20,
                                 validation_split=0.2,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.5,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 brightness_range=(0.8, 1.2),
                                 #channel_shift_range=0.9,
                                 vertical_flip=True,
                                 preprocessing_function=return_preprocessing_function(type_of_model))
    datagen.fit(x_train)
    train_generator = datagen.flow(x_train, y_train, batch_size=batch_size, subset='training')
    val_generator = datagen.flow(x_train, y_train, batch_size=batch_size, subset='validation')
    return train_generator, val_generator


def save_generated_batch(generator, batch_number, path="batch_images"):
    i = 0
    p = Path(path)
    if(not p.exists()):
        p.mkdir(parents=True)
    for X_batch, y_batch in generator:
        if (i == batch_number):
            f = figure(figsize=(20, 20))
            for i in range(0, batch_size):
                fig = f.add_subplot(batch_size // 8, 8, i + 1)
                fig.set_title(CLASSES[y_batch[i].tolist().index(1.0)])
                tight_layout()
                imshow(X_batch[i].reshape(HEIGHT, WIDTH, 3))
            savefig((p / ("batch" + str(batch_number))))
            return
        i += 1
