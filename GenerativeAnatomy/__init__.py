import sys
from computer import computer
import os

if computer == 'desktop':
    # path_df = './training_info_Mar.5.2022.csv'
    OP_CACHE_DIR = '/home/gattia/data/notebooks/stanford/diffusion_net/data/op_cache'
    sys.path.append("/home/gattia/programming/diffusion-net/src/")
elif computer == 'server':
    # path_df = '/bmrNAS/people/aagatti/projects/Diffusion_Net/training_info_Feb.21.2022.csv'
    OP_CACHE_DIR = '/bmrNAS/people/aagatti/projects/Diffusion_Net/notebooks/diffusion_net/data/op_cache'
    sys.path.append('/bmrNAS/people/aagatti/projects/Diffusion_Net')
    sys.path.append('/dataNAS/people/aagatti/programming/diffusion-net/src/')

os.environ['OP_CACHE_DIR'] = OP_CACHE_DIR

from .main import *
from . import datasets
from . import models               