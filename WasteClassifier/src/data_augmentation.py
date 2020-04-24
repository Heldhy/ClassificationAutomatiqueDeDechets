import preprocessing
from variables import *
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_generator(x_train, y_train):
    datagen = ImageDataGenerator(rotation_range=20,
                                 validation_split=0.2,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.5,
                                 zoom_range=(0.9, 1.1),
                                 preprocessing_function=preprocessing.preprocess_input)
    datagen.fit(x_train)
    train_generator = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, subset='training')
    val_generator = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, subset='validation')
    return train_generator, val_generator


def print_generated_batch(generator, batch_number):
    i = 0
    for X_batch, y_batch in generator:
        if (i == batch_number):
            f = plt.figure(figsize=(20, 20))
            for i in range(0, BATCH_SIZE):
                fig = f.add_subplot(BATCH_SIZE // 8, 8, i + 1)
                fig.set_title(classes[y_batch[i].tolist().index(1.0)])
                plt.tight_layout()
                plt.imshow(X_batch[i].reshape(HEIGHT, WIDTH, 3))
            plt.show()
            return
        i += 1
