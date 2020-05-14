from pathlib import Path

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
WASTE_TYPE = ['recyclable', 'verre', 'non recyclable']
CLASSES_TO_TRASH = {'cardboard': 0, 'glass': 1, 'metal': 0, 'paper': 0, 'plastic': 0, 'trash': 2}
CLASS_TO_INDEX = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5}
NB_CLASSES = 6
BASE_DIR = Path('.') / 'data' / "dataset-resized"
WIDTH = 224
HEIGHT = 224
batch_size = 32
epoch = 30
filepath = "bestmodel.h5"
model_type = "mobilenet"
optimizer_type = "rmsprop"


from .pre_processing import *
from .choose_model import *
from .data_augmentation import *
from .training import *
from .post_processing import *
from .use_model import *