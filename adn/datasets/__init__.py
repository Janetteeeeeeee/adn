import os.path as path
from .deep_lesion import DeepLesion
from .spineweb import Spineweb
from .nature_image import NatureImage
from .brachy_ct import BrachyCT

def get_dataset(dataset_type, **dataset_opts):
    return {
        "deep_lesion": DeepLesion,
        "spineweb": Spineweb,
        "nature_image": NatureImage,
        "brachy_ct": BrachyCT
    }[dataset_type](**dataset_opts[dataset_type])
