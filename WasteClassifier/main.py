import data_augmentation
import post_processing
import pre_processing
import training


def main():
    x_train, y_train, x_test, y_test = pre_processing.preprocessing()
    train_generator, val_generator = data_augmentation.create_generator(x_train, y_train)
    data_augmentation.save_generated_batch(train_generator, 2)
    model = training.training(train_generator, val_generator, x_test, y_test, True)
    post_processing.evaluate_post_processed_prediction(model, x_test, y_test)


main()
