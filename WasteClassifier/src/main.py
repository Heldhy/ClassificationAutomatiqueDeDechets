from variables import *
import preprocessing
import postprocessing
import training
import data_augmentation


def main():
    x_train, y_train, x_test, y_test = preprocessing.preprocessing()
    train_generator, val_generator = data_augmentation.create_generator(x_train, y_train)
    model = training.training(train_generator, val_generator, x_test, y_test, True)
    postprocessing.evaluate_postprocessed_model(model, x_test, y_test)


main()