import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
import numpy as np
from evaluation import get_cleaned_processed_df, drop_text_features

LOG_DIR='log'

def visualize_data(df):
    df = drop_text_features(df)
    df.fillna("", inplace=True)

    df.to_csv(os.path.join(LOG_DIR, 'metadata.tsv'), sep='\t', mode='w+')

    training_labels = df["label"].values
    df = df.drop("label",1)
    training_data = df.values

    input_var_data = tf.Variable(initial_value=training_data, dtype=np.dtype(float))
    input_var_labels = tf.Variable(initial_value=training_labels)

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    summary_writer = tf.train.SummaryWriter(LOG_DIR)

    # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = input_var_data.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

    # Saves a configuration file that TensorBoard will read during startup.
    projector.visualize_embeddings(summary_writer, config)

    with tf.Session() as sess:

        sess.run(init_op)

        for n in range(1):
            saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), n)



if __name__ == '__main__':
   visualize_data(get_cleaned_processed_df())