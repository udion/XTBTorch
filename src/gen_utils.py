import os, sys, random
import numpy as np
import PIL
from PIL import Image

def get_lbl_from_name(fname):
    lbl = int(fname.split('.png')[0][-1])
    return lbl