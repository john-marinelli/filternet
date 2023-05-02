from signalnet import utilities
from tensorflow import keras


def create_lstm(lstm_dim=128, dense_dims=[256, 32], output_dim=1, input_shape=(5, 1),
                lstm_activation="tanh", dense_activations=["relu", "tanh"]):

    model = keras.models.Sequential()

    model.add(keras.layers.LSTM(
        lstm_dim, input_shape=input_shape, activation=lstm_activation))

    for idx, dim in enumerate(dense_dims):
        model.add(keras.layers.Dense(dim, activation=dense_activations[idx]))

    model.add(keras.layers.Dense(output_dim))

    return model
