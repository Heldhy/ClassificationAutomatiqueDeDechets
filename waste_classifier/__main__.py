from numpy import argmax
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical

from calibration import calibrate_on_test, get_logits_friendly_model, calibrate_on_validation
from data_augmentation import create_new_generator
from pre_processing import get_un_pre_processed_test_data, pre_processing
from soft_recall import soft_recall_from_model
from training import training_with_fine_tuning
from waste_classifier import filepath, filepath_logit, CLASSES, AN_TABLE


def main():
    x_train, y_train, x_test, y_test = pre_processing()
    """
    train_generator, val_generator = create_new_generator(x_train, y_train)
    model, history = training_with_fine_tuning(train_generator, val_generator, x_test, y_test)
    model.save(filepath)
    """
    model = load_model(filepath)
    res = soft_recall_from_model(model, x_test, y_test, AN_TABLE, CLASSES)
    print(res)
    """
    logit_model = get_logits_friendly_model(model)
    logit_model.save(filepath_logit)

    temp = calibrate_on_validation(filepath, val_generator)
    second_temp = calibrate_on_test(filepath, x_test, y_test)
    """


if __name__ == "__main__":
    main()
