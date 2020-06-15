from tensorflow.python.keras.models import load_model

from waste_classifier import pre_processing, create_new_generator, training_with_fine_tuning, \
    evaluate_post_processed_prediction, filepath, soft_recall_from_model, AN_TABLE, CLASSES, calibrate_model, \
    get_logits_friendly_model, filepath_logit, reliability_diagram


def main():
    x_train, y_train, x_test, y_test = pre_processing()
    train_generator, val_generator = create_new_generator(x_train, y_train)
    # model, history = training_with_fine_tuning(train_generator, val_generator, x_test, y_test)
    model = load_model(filepath)

    temp = calibrate_model(filepath, val_generator)

    model.save(filepath)
    logit_model = get_logits_friendly_model(model)
    logit_model.save(filepath_logit)


if __name__ == "__main__":
    main()
