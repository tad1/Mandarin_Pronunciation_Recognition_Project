from math import nan
import math
import tensorflow as tf
from path import RESULT_DIRECTORY

def get_tf_dataset(size : int, use_cache = True, use_minimal_duration=True) -> tf.data.Dataset:
    dataset_name = f"tf_{size}_dataset"
    dataset_dir = RESULT_DIRECTORY
    dataset_path = f"{dataset_dir}/{dataset_name}"
    if use_cache:
        if tf.io.gfile.exists(dataset_path):
            print(f"Loading dataset from {dataset_path}")
            dataset = tf.data.Dataset.load(dataset_path)
            return dataset
    
    print(f"Creating dataset of size {size}")
    audio, tone_sequences = get_dataset(size, use_minimal_duration)
    max_lenght = max([len(sequence) for sequence in audio])

    print("converting dataset")
    audio = [[val if not math.isnan(val) else 0 for val in sequence] for sequence in audio]
    tone_sequences = [[val.value for val in sequence] for sequence in tone_sequences]
    audio = tf.ragged.constant(audio)
    audio = audio.to_tensor()
    audio = tf.expand_dims(audio, axis=-1)
    
    tone_sequences = tf.ragged.constant(tone_sequences)
    tone_sequences = tone_sequences.to_tensor()

    dataset = tf.data.Dataset.from_tensor_slices((audio, tone_sequences))
        # tf.io.gfile.makedirs(dataset_dir)
    dataset.save(dataset_path)
    return dataset

if __name__ == "__main__":
    dataset = get_tf_dataset(1000, use_cache=False)
    for audio, tones in dataset.take(5):
        print(audio.shape)
        print(tones)