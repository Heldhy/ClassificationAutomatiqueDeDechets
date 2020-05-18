from waste_classifier import pre_processing, create_new_generator, save_generated_batch, training, \
    evaluate_post_processed_prediction, training_extractor, training_from_extracted_features


def main():
    x_train, y_train, x_test, y_test = pre_processing()
    model = training_from_extracted_features("bottleneck_features_train.npy", y_train, None, "bottleneck_features_test.npy", y_test, True)
    #evaluate_post_processed_prediction(model, x_test, y_test)


if __name__ == "__main__":
    main()
