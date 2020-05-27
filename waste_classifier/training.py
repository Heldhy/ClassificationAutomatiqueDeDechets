import tensorflow as tf
from numpy import argmax, save, load
from sklearn.metrics import recall_score, accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop

from waste_classifier import filepath, optimizer_type, model_type, epoch, batch_size, get_model, \
    bootleneck_feature_extractor, create_only_one_generator_for_feature_extraction, no_pre_processing, NB_CLASSES, \
    build_hybrid_model, return_frozen_mobilenet


def extract_train_features(x_train, y_train):
    extractor = bootleneck_feature_extractor()
    data_generator = create_only_one_generator_for_feature_extraction(x_train, y_train)
    features = extractor.predict_generator(data_generator, len(x_train) // batch_size)
    return features, data_generator


def extract_test_features(x_test):
    extractor = bootleneck_feature_extractor()
    features = extractor.predict(x_test)
    return features


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


def get_optimizer(optimizer_type, lr=0.00005):
    if (optimizer_type == "rmsprop"):
        return RMSprop(lr=lr)
    return None


def training_from_extracted_features(path_to_train_features, y_train, chosen_model=None, path_to_test_features=None, y_test=None, evaluate=False):
    if (chosen_model is None):
        chosen_model = model_type
    train_data = load(path_to_train_features)
    nb_train_sample = len(train_data)
    y_train_processed = argmax(y_train[:nb_train_sample], axis = 1)
    model = get_model(chosen_model)
    model.fit(train_data.reshape(nb_train_sample, -1), y_train_processed)
    if (evaluate and path_to_test_features is not None and y_test is not None):
        test_data = load(path_to_test_features)
        nb_test_sample = len(test_data)
        y_pred = model.predict(test_data.reshape((nb_test_sample, -1)))
        y_test_processed = argmax(y_test, axis = 1)
        recall = recall_score(y_test_processed, y_pred, average='macro')
        accuracy = accuracy_score(y_test_processed, y_pred)
        print("accuracy : " + str(accuracy))
        print("recall : " + str(recall))
    return model


def training_extractor(x_train, y_train, x_test, y_test, chosen_model=None, evaluate=False, saving=False):
    if (chosen_model is None):
        chosen_model = model_type
    model = get_model(chosen_model)
    features, data_generator = extract_train_features(x_train, y_train)
    if(saving):
        save('bottleneck_features_train.npy', features)
    nb_train_sample = len(features)
    y_train = data_generator.y[:nb_train_sample]
    y_train = argmax(y_train, axis = 1)
    features_reshaped = features.reshape((nb_train_sample, -1))
    model.fit(features_reshaped, y_train)
    if(evaluate):
        test_features = extract_test_features(x_test)
        if(saving):
            save('bottleneck_features_test.npy', test_features)
        y_pred = model.predict(test_features.reshape(len(test_features), -1))
        y_test_converted = argmax(y_test, axis = 1)
        recall = recall_score(y_test_converted, y_pred, average='macro')
        accuracy = accuracy_score(y_test_converted, y_pred)
        print("accuracy : " + str(accuracy))
        print("recall : " + str(recall))
    return model


def fine_tuning(train_generator, val_generator, x_test, y_test, evaluate=False):
    base_model = return_frozen_mobilenet()
    model = build_hybrid_model(base_model)
    callbacks = create_callbacks_list()
    optimizer = get_optimizer(optimizer_type, 0.0005)
    compile_model(model, optimizer)
    print(model.summary())
    fit(model, train_generator, val_generator, callbacks, 20)

    model = load_model(filepath)
    base_model = model.layers[0]
    base_model.trainable = True

    fine_tune_at = 20
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    optimizer = get_optimizer(optimizer_type)
    compile_model(model, optimizer, 0.00001)
    history = fit(model, train_generator, val_generator, callbacks, 70)
    model = load_model(filepath)

    if (evaluate):
        score = evaluate_model(model, x_test, y_test)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        print('Test recall:', score[2])
    return model, history


def training(train_generator, val_generator, x_test, y_test, chosen_model=None, evaluate=False):
    if(chosen_model is None):
        chosen_model = model_type
    model = get_model(chosen_model)
    callbacks = create_callbacks_list()
    optimizer = get_optimizer(optimizer_type)
    compile_model(model, optimizer)
    history = fit(model, train_generator, val_generator, callbacks, epoch)
    model = load_model(filepath)
    if (evaluate):
        score = evaluate_model(model, x_test, y_test)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        print('Test recall:', score[2])
    return model, history
