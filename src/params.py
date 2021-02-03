import warnings
import numpy as np

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

SEED = 2020

LOG_PATH = "../logs/"
DATA_PATH = "../data/"
OUT_DIR = "../output/"


MELS_PATH = "../data/new/"

TRAIN_MELS_PATH = MELS_PATH + "train/"
TEST_MELS_PATH = MELS_PATH + "test/"

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

NUM_WORKERS = 4

NUM_CLASSES = 24
NUM_FEATURES = 128
SPEC_LENGTH = 3001
SCALES = [32, 64, 128, 192, 256, 320, 384, 448, 512]


HOP_SIZE_RATIO = 50
NUM_FEATURES = 128
