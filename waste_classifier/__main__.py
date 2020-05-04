from waste_classifier import pre_processing, create_generator, save_generated_batch, training, \
    evaluate_post_processed_prediction


def main():
    x_train, y_train, x_test, y_test = pre_processing()
    train_generator, val_generator = create_generator(x_train, y_train)
    save_generated_batch(train_generator, 2)
    model = training(train_generator, val_generator, x_test, y_test, True)
    evaluate_post_processed_prediction(model, x_test, y_test)


if __name__ == "__main__":
    main()
