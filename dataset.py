import random
import numpy as np
from tensorflow import keras
from config import *


class AnchorPositivePairs(keras.utils.Sequence):
    def __init__(self, num_batchs, x_train, class_idx_to_train_idx):
        self.num_batchs = num_batchs
        self.x_train = x_train
        self.class_idx_to_train_idxs = class_idx_to_train_idx

    def __len__(self):
        return self.num_batchs

    def __getitem__(self, _idx):
        x = np.empty((2, num_classes, height_width, height_width, 3), dtype=np.float32)
        for class_idx in range(num_classes):
            examples_for_class = self.class_idx_to_train_idxs[class_idx]
            anchor_idx = random.choice(examples_for_class)
            positive_idx = random.choice(examples_for_class)
            while positive_idx == anchor_idx:
                positive_idx = random.choice(examples_for_class)
            x[0, class_idx] = self.x_train[anchor_idx]
            x[1, class_idx] = self.x_train[positive_idx]
        return x
