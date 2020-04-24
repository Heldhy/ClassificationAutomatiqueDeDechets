import models
import preprocessing
import data_augmentation
from variables import *
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model


def create_callbacks_list():
    checkpoint = ModelCheckpoint(filepath, monitor='val_recall', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    return callbacks_list


def compile_model(model, optimizer):
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Recall(name="recall")]
                  )


def fit(model, train_generator, val_generator, callbacks, epoch):
    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.n // BATCH_SIZE,
                        validation_steps=val_generator.n // BATCH_SIZE,
                        epochs=epoch,
                        validation_data=val_generator,
                        callbacks=callbacks,
                        use_multiprocessing=False)


def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    return score


def get_optimizer(optimizer_type):
    if (optimizer_type == "rmsprop"):
        return RMSprop(lr=0.00005)
    return None


def training(train_generator, val_generator, x_test, y_test, evaluate=False):
    callbacks = create_callbacks_list()
    optimizer = get_optimizer(optimizer_type)
    model = models.get_model(modeltype)
    compile_model(model, optimizer)
    fit(model, train_generator, val_generator, callbacks, epoch)
    del model
    model = load_model(filepath)
    if (evaluate):
        score = evaluate_model(model, x_test, y_test)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        print('Test recall:', score[2])
    return model
