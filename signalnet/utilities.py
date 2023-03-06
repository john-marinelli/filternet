import numpy as np


def frame_up(X_train, X_test, y_train, y_test, window=9):
    half_window = int(window / 2)
    X_train_ann = np.zeros((X_train.shape[0] - (window-1), window))
    y_train_ann = y_train[half_window:-half_window].copy()

    for idx, item in enumerate(X_train[half_window:-half_window]):
        X_train_ann[idx] = [X_train[idx+(i-half_window)]
                            for i in range(window)]

    X_test_ann = np.zeros((X_test.shape[0] - (window-1), window))
    y_test_ann = y_test[half_window:-half_window]

    for idx, item in enumerate(X_test[half_window:-half_window]):
        X_test_ann[idx] = [X_test[idx+(i-half_window)] for i in range(window)]

    return X_train_ann, X_test_ann, y_train_ann, y_test_ann


def train_test(X, y, percent=None):
    length = len(y)
    mid = int(length / 2)
    return X[:mid], X[mid:], y[:mid], y[mid:]
