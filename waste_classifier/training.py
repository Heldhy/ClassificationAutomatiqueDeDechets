import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop

from waste_classifier import filepath, batch_size, add_classification_layer, return_frozen_mobilenet


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


def training_with_fine_tuning(train_generator, val_generator, x_test, y_test, evaluate=False):
    base_model = return_frozen_mobilenet()
    model = add_classification_layer(base_model)
    callbacks = create_callbacks_list()
    optimizer = get_optimizer(0.0005)
    compile_model(model, optimizer)
    fit(model, train_generator, val_generator, callbacks, 25)

    model = load_model(filepath)
    base_model = model.layers[0]
    base_model.trainable = True

    optimizer = get_optimizer(0.00005)
    compile_model(model, optimizer)
    history = fit(model, train_generator, val_generator, callbacks, 100)
    model = load_model(filepath)

    if (evaluate):
        score = evaluate_model(model, x_test, y_test)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        print('Test recall:', score[2])
    return model, history
