import tensorflow as tf
import numpy as np
import time
# I got to wonder how well CTC loss scales.

# Function to generate random sequences and labels
def generate_data(batch_size, max_seq_length, num_labels, max_label_length):
    sequences = tf.random.uniform((batch_size, max_seq_length, num_labels), dtype=tf.float32)
    labels = tf.random.uniform((batch_size, max_label_length), minval=1, maxval=num_labels, dtype=tf.int32)
    input_lengths = tf.fill([batch_size], max_seq_length)
    label_lengths = tf.random.uniform([batch_size], minval=1, maxval=max_label_length + 1, dtype=tf.int32)
    return sequences, labels, input_lengths, label_lengths

# Performance test for CTC loss without LSTM model
def test_ctc_loss_without_lstm():
    num_labels = 5
    max_seq_length = 100
    max_label_length = 20
    test_cases = [10, 100, 200, 500]
    ctc_loss = tf.keras.losses.CTC(reduction="none")

    print("CTC Loss Performance Test without LSTM Model")
    
    for num_sequences in test_cases:
        for _ in range(2):  # Repeat each scenario two times
            sequences, labels, input_lengths, label_lengths = generate_data(10, num_sequences, num_labels, max_label_length)
            print(sequences.shape)

            # Compute CTC loss
            start_time = time.time()
            
            loss = ctc_loss(labels, sequences)
            elapsed_time = time.time() - start_time
            
            print(f"Num sequences: {num_sequences}, Time: {elapsed_time:.4f} seconds")

# Run the performance test without LSTM
test_ctc_loss_without_lstm()
