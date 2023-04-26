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


def frame_up_new(X_train, X_test, y_train, y_test, window=9):
    X_train_new = []
    y_train_new = []
    X_test_new = []
    y_test_new = []

    for idx, item in enumerate(X_train):
        if idx >= window and idx < (len(y_train) - 1):
            frame = list(X_train[idx-window:idx])
            X_train_new.append(frame.copy())
            frame.append(y_train[idx+1])
            y_train_new.append(frame.copy())

    for idx, item in enumerate(X_test):
        if idx >= window and idx < (len(y_test) - 1):
            frame = list(X_test[idx-window:idx])
            X_test_new.append(frame.copy())
            frame.append(y_test[idx+1])
            y_test_new.append(frame.copy())
    print(len(X_train_new[0]))
    print(np.array(X_train_new).shape)
    return np.array(X_train_new), np.array(X_test_new), np.array(y_train_new), np.array(y_test_new)


def train_test(X, y, percent=None):
    length = len(y)
    mid = int(length / 2)
    return X[:mid], X[mid:], y[:mid], y[mid:]
