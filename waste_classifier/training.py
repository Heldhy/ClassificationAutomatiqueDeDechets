from pathlib import Path

import tensorflow as tf
from matplotlib.pyplot import subplots, savefig
from numpy.ma import arange
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop

from waste_classifier import filepath, batch_size, return_frozen_mobilenet, add_classification_layer


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
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_generator.n // batch_size,
                                  validation_steps=val_generator.n // batch_size,
                                  epochs=epoch,
                                  validation_data=val_generator,
                                  callbacks=callbacks,
                                  use_multiprocessing=False)
    return history


def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    return score


def get_optimizer(lr=0.00005):
    return RMSprop(lr=lr)


def training_visualisation(history, nb_epoch, title, path="learning_curves"):
    nb_epoch += 1
    path_to_save_at = Path(path)
    if (not path_to_save_at.exists()):
        path_to_save_at.mkdir(parents=True)
    figure_plot, (ax1, ax2, ax3) = subplots(1, 3, figsize=(20, 6))
    figure_plot.suptitle('MobileNet Performances', fontsize=12)
    figure_plot.subplots_adjust(top=0.85, wspace=0.3)

    epoch_list = list(range(1, nb_epoch))
    ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
    ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xticks(arange(0, nb_epoch, 5))
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    ax1.legend(loc="best")

    ax2.plot(epoch_list, history.history['recall'], label='Train Recall')
    ax2.plot(epoch_list, history.history['val_recall'], label='Validation Recall')
    ax2.set_xticks(arange(0, nb_epoch, 5))
    ax2.set_ylabel('Recall Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Recall')
    ax2.legend(loc="best")

    ax3.plot(epoch_list, history.history['loss'], label='Train Loss')
    ax3.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
    ax3.set_xticks(arange(0, nb_epoch, 5))
    ax3.set_ylabel('Loss Value')
    ax3.set_xlabel('Epoch')
    ax3.set_title('Loss')
    ax3.legend(loc="best")
    savefig(path_to_save_at / title)


def training_with_fine_tuning(train_generator, val_generator, x_test, y_test, evaluate=False):
    base_model = return_frozen_mobilenet()
    model = add_classification_layer(base_model)
    callbacks = create_callbacks_list()
    optimizer = get_optimizer(0.0005)
    compile_model(model, optimizer)
    history = fit(model, train_generator, val_generator, callbacks, 25)
    training_visualisation(history, 25, "training")

    model = load_model(filepath)
    base_model = model.layers[0]
    base_model.trainable = True

    optimizer = get_optimizer(0.00005)
    compile_model(model, optimizer)
    history = fit(model, train_generator, val_generator, callbacks, 100)
    training_visualisation(history, 100, "fine_tuning")
    model = load_model(filepath)

    if (evaluate):
        score = evaluate_model(model, x_test, y_test)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        print('Test recall:', score[2])
    return model, history
