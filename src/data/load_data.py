import os
import pandas as pd

from config import RAW_TRAIN_PATH, RAW_TEST_PATH

def load_train_data():
    train_raw = pd.read_csv(RAW_TRAIN_PATH)
    return train_raw

def load_test_data():
    test_raw = pd.read_csv(RAW_TEST_PATH)
    return test_raw


if __name__ == "__main__":
    load_train_data()
    load_test_data()
