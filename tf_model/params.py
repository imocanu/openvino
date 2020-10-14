import os
import time
import numpy as np
import random
import tensorflow as tf

WSIZE = 32  # window size
LOOKUP = 1

TSIZE = 0.4 # test/train split

DFCOLUMNS = ["Adj Close", "Volume", "Open", "High", "Low"]

BATCH_SIZE = 3
EPOCHS = 500

OPTIMIZER="adam"
LOSS="mae"

tf.random.set_seed(112)
np.random.seed(112)
random.seed(112)