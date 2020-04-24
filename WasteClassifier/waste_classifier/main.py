from waste_classifier import training, data_augmentation, post_processing, pre_processing


def main():
    x_train, y_train, x_test, y_test = pre_processing.preprocessing()
    train_generator, val_generator = data_augmentation.create_generator(x_train, y_train)
    model = training.training(train_generator, val_generator, x_test, y_test, True)
    post_processing.evaluate_postprocessed_model(model, x_test, y_test)


main()