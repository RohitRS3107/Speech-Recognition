import tensorflow as tf
from tensorflow import keras
import pandas as pd

def get_datasets(metadata_path, wavs_path, batch_size=32):
    metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
    metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
    metadata_df = metadata_df[["file_name", "normalized_transcription"]].sample(frac=1)

    split = int(len(metadata_df) * 0.70)
    df_train, df_val = metadata_df[:split], metadata_df[split:]

    characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
    char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
    num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

    def encode_sample(wav_file, label):
        file = tf.io.read_file(wavs_path + wav_file + ".wav")
        audio, _ = tf.audio.decode_wav(file)
        audio = tf.cast(tf.squeeze(audio, axis=-1), tf.float32)
        spectrogram = tf.abs(tf.signal.stft(audio, frame_length=256, frame_step=160, fft_length=384))
        spectrogram = tf.math.pow(spectrogram, 0.5)
        means, stddevs = tf.math.reduce_mean(spectrogram, 1, keepdims=True), tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)
        label = char_to_num(tf.strings.unicode_split(tf.strings.lower(label), "UTF-8"))
        return spectrogram, label

    train_ds = tf.data.Dataset.from_tensor_slices((list(df_train["file_name"]), list(df_train["normalized_transcription"])))
    train_ds = train_ds.map(encode_sample, num_parallel_calls=tf.data.AUTOTUNE).padded_batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((list(df_val["file_name"]), list(df_val["normalized_transcription"])))
    val_ds = val_ds.map(encode_sample, num_parallel_calls=tf.data.AUTOTUNE).padded_batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, char_to_num, num_to_char
