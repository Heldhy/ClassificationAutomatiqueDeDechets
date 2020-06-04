from waste_classifier import pre_processing, create_new_generator, training_with_fine_tuning, \
    evaluate_post_processed_prediction


def main():
    x_train, y_train, x_test, y_test = pre_processing()
    train_generator, val_generator = create_new_generator(x_train, y_train)
    model, history = training_with_fine_tuning(train_generator, val_generator, x_test, y_test)
    evaluate_post_processed_prediction(model, x_test, y_test)


if __name__ == "__main__":
    main()
