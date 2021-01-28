import torch
import warnings
import numpy as np

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

SEED = 2020

LOG_PATH = "../logs/"
DATA_PATH = "../input/"
OUT_DIR = "../output/"


MELS_PATH = "../../../data/rcfx/new/"

TRAIN_MELS_PATH = MELS_PATH + "train/"
TEST_MELS_PATH = MELS_PATH + "test/"

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

NUM_WORKERS = 4

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

NUM_CLASSES = 24
