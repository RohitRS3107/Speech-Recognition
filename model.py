import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones(shape=(batch_len, 1), dtype="int64")
    return keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def build_model(input_dim, output_dim):
    input_spectrogram = layers.Input((None, input_dim), name="input")
    x = layers.Reshape((-1, input_dim, 1))(input_spectrogram)
    # 2D CNN layers
    x = layers.Conv2D(32, [11, 41], strides=[2, 2], padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, [11, 21], strides=[1, 2], padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    # 5 RNN layers
    for i in range(5):
        recurrent = layers.GRU(512, activation="tanh", return_sequences=True, reset_after=True)
        x = layers.Bidirectional(recurrent, merge_mode="concat")(x)
        if i < 4: x = layers.Dropout(0.5)(x)
    output = layers.Dense(output_dim + 1, activation="softmax")(x)
    model = keras.Model(input_spectrogram, output)
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=CTCLoss)
    return model
