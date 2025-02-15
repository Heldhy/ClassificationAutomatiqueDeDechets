from pathlib import Path

from matplotlib.pyplot import figure, tight_layout, savefig, plot
from numpy import argmax, zeros, ndarray
from sklearn.calibration import calibration_curve
from tensorflow.python import convert_to_tensor, float32, int32, Variable, reduce_mean, divide
from tensorflow.python.keras.applications.mobilenet import preprocess_input
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.ops.nn_ops import softmax, softmax_cross_entropy_with_logits_v2
from tensorflow_probability.python.stats import expected_calibration_error

from waste_classifier.training import compile_model
from waste_classifier import NB_CLASSES, CLASSES


def from_multiclass_to_one_vs_all(y_test: ndarray, considerate_class: int) -> ndarray:
    y_res = zeros((len(y_test), 1))
    for i in range(len(y_test)):
        if argmax(y_test[i]) == considerate_class:
            y_res[i] = 1
        else:
            y_res[i] = 0
    return y_res


def reliability_diagram(prediction: ndarray, y: ndarray, name: str, path: str = 'reliability_diagram'):
    path_to_save_at = Path(path)
    if not path_to_save_at.exists():
        path_to_save_at.mkdir(parents=True)
    figure_plot = figure(figsize=(20, 20))
    for considerate_class in range(NB_CLASSES):
        probs = prediction[:, considerate_class]
        y_one_vs_all = from_multiclass_to_one_vs_all(y, considerate_class)
        fop, mpv = calibration_curve(y_one_vs_all, probs, n_bins=10)
        fig = figure_plot.add_subplot(NB_CLASSES // 2, 2, considerate_class + 1)
        fig.set_title(CLASSES[considerate_class])
        tight_layout()
        plot([0, 1], [0, 1], linestyle='--')
        plot(mpv, fop, marker='.')
        savefig(path_to_save_at / name)


def get_logits_friendly_model(model):
    model.layers[1].activation = None
    optimizer = model.optimizer
    compile_model(model, optimizer)
    return model


def compute_ECE(logit_model, x: ndarray, y: ndarray, num_bins: int, temperature_scaling: float = 1.0) -> float:
    prediction = logit_model.predict(x)
    prediction_scaled = prediction / temperature_scaling
    logits = convert_to_tensor(prediction_scaled, dtype=float32, name='logits')
    labels_true = convert_to_tensor(argmax(y, axis=1), dtype=int32, name='labels_true')
    ece = expected_calibration_error(num_bins=num_bins, logits=logits, labels_true=labels_true)
    return ece.numpy()


def compute_temperature_scaling(logit_model, x: ndarray, y: ndarray) -> float:
    temp = Variable(initial_value=1.0, trainable=True, dtype=float32)
    logits = logit_model.predict(x)

    def compute_loss():
        divided_prediction = divide(logits, temp)
        loss = reduce_mean(
            softmax_cross_entropy_with_logits_v2(labels=convert_to_tensor(y), logits=divided_prediction))
        return loss

    optimizer = Adam(learning_rate=0.01)
    for i in range(1000):
        optimizer.minimize(compute_loss, var_list=[temp])
    return temp.numpy()


def calibrate(logit_model, x: ndarray, y: ndarray) -> float:
    uncalibrated_logits = logit_model.predict(x)
    uncalibrated_prediction = softmax(uncalibrated_logits)

    print("ECE before calibration: " + str(compute_ECE(logit_model, x, y, 20, 1)))
    reliability_diagram(uncalibrated_prediction, y, "before_calibration")

    temperature_scaling = compute_temperature_scaling(logit_model, x, y)

    calibrated_logits = uncalibrated_logits / temperature_scaling
    calibrated_prediction = softmax(calibrated_logits)

    print("scaling by " + str(temperature_scaling))
    print("ECE after calibration: " + str(compute_ECE(logit_model, x, y, 20, temperature_scaling)))
    reliability_diagram(calibrated_prediction, y, "after_calibration")
    return temperature_scaling


def calibrate_on_validation(model_path: str, val_generator) -> float:
    logit_model = get_logits_friendly_model(load_model(model_path))
    x_val = val_generator.x
    y_val = val_generator.y
    x_val_pre = preprocess_input(x_val)
    return calibrate(logit_model, x_val_pre, y_val)


def calibrate_on_test(model_path: str, x_test: ndarray, y_test: ndarray) -> float:
    logit_model = get_logits_friendly_model(load_model(model_path))
    return calibrate(logit_model, x_test, y_test)
