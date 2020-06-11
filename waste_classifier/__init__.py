from pathlib import Path

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
WASTE_TYPE = ['recyclable', 'verre', 'non recyclable']
CLASSES_TO_TRASH = {'cardboard': 0, 'glass': 1, 'metal': 0, 'paper': 0, 'plastic': 0, 'trash': 2}
CLASS_TO_INDEX = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5}
NB_CLASSES = 6
BASE_DIR = Path('.') / 'data' / "dataset-resized"
WIDTH = 224
HEIGHT = 224
batch_size = 64
filepath = "bestmodel.h5"

from .pre_processing import make_square, get_preprocessed_data, get_data, pre_processing
from .data_augmentation import create_new_generator, save_generated_batch
from .choose_model import return_frozen_mobilenet, add_classification_layer
from .training import create_callbacks_list, compile_model, fit, evaluate_model, get_optimizer, \
    training_visualisation, training_with_fine_tuning
from .use_model import return_trash_label, predict_image
from .post_processing import convert_to_trash, predict_and_convert_to_trash, evaluate_post_processed_prediction
