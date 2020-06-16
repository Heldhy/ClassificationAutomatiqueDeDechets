from pathlib import Path

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
WASTE_TYPE = ['recyclable', 'verre', 'non recyclable']
CLASSES_TO_TRASH = {'cardboard': 0, 'glass': 1, 'metal': 0, 'paper': 0, 'plastic': 0, 'trash': 2}
CLASS_TO_INDEX = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5}
NB_CLASSES = 6
AN_TABLE = {"paper": ["plastic", "cardboard", "metal", "trash"], "plastic": ["paper", "cardboard", "metal", "trash"],
            "cardboard": ["paper", "plastic", "metal", "trash"], "metal": ["paper", "plastic", "cardboard", "trash"],
            "glass": ["trash"], "trash": []}
AN_PRO_PROCESS_TALE = {'recyclable': ['non recyclable'], 'verre':['non recyclable'], 'non recyclable':[]}
BASE_DIR = Path('.') / 'data' / "dataset-resized"
WIDTH = 224
HEIGHT = 224
batch_size = 64
filepath = "bestmodel.h5"
filepath_logit = "bestmodel_logits.h5"


from .pre_processing import make_image_square, get_preprocessed_data, get_data, pre_processing
from .data_augmentation import create_new_generator, save_generated_batch
from .choose_model import return_frozen_mobilenet, add_classification_layer
from .training import create_callbacks_list, compile_model, fit, evaluate_model, get_optimizer, \
    training_visualisation, training_with_fine_tuning
from .calibration import reliability_diagram, compute_ECE, get_logits_friendly_model, \
    compute_temperature_scaling, calibrate_model, reliability_diagram_from_model, calibrate_on_test
from .use_model import return_trash_label, predict_image
from .post_processing import convert_to_trash, predict_and_convert_to_trash, evaluate_post_processed_prediction
from .soft_recall import soft_recall_function, soft_recall_from_model

