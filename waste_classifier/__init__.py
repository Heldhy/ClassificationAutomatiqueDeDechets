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
filepath_logit = "logits_model.h5"

