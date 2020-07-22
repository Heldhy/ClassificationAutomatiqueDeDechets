import tensorflow as tf
from tensorflow.python.keras.models import load_model

from waste_classifier import filepath_logit
from waste_classifier.post_processing import compute_best_threshold_for_dispatching_of_trash, \
    compute_trash_classification_with_threshold
from waste_classifier.pre_processing import pre_processing


def main():
    x_train, y_train, x_test, y_test = pre_processing()

    # train_generator, val_generator = create_new_generator(x_train, y_train)
    # model, history = training_with_fine_tuning(train_generator, val_generator, x_test, y_test)
    # model.save(filepath)
    # res = soft_recall_from_model(model, x_test, y_test, AN_TABLE, CLASSES)
    # logit_model = get_logits_friendly_model(model)
    # logit_model.save(filepath_logit)
    # temp = calibrate_on_validation(filepath, val_generator)
    # second_temp = calibrate_on_test(filepath, x_test, y_test)

    model = load_model(filepath_logit)
    with open('temperature_scaling.txt', 'r') as f:
        temperature = float(f.readline())
    prediction = tf.nn.softmax(model.predict(x_test) / temperature).numpy()
    score = compute_best_threshold_for_dispatching_of_trash(prediction, y_test)
    print("Before thresholding")
    compute_trash_classification_with_threshold(prediction, y_test, 0)
    print("After thresholding with threshold " + str(score))
    compute_trash_classification_with_threshold(prediction, y_test, score)


if __name__ == "__main__":
    main()
