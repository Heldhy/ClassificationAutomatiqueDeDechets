from waste_classifier import pre_processing, create_new_generator, save_generated_batch, training, \
    evaluate_post_processed_prediction, training_extractor, training_from_extracted_features, fine_tuning


def main():
    x_train, y_train, x_test, y_test = pre_processing()
    train_generator, val_generator = create_new_generator(x_train, y_train, "mobilenet")
    model, history = fine_tuning(train_generator, val_generator, x_test, y_test, True)
    #evaluate_post_processed_prediction(model, x_test, y_test)


if __name__ == "__main__":
    main()
