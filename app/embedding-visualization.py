import tensorflow as tf
import os
import numpy as np
from evaluation import get_cleaned_processed_df, drop_text_features

LOG_DIR='log'

def visualize_data(df):
    df = drop_text_features(df)
    df.fillna("", inplace=True)

    training_labels = df["label"].values
    df = df.drop("label",1)
    training_data = df.values

    input_var_data = tf.Variable(initial_value=training_data, dtype=np.dtype(float))
    input_var_labels = tf.Variable(initial_value=training_labels)

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    with tf.Session() as sess:

        sess.run(init_op)

        for n in range(1):
            saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), n)



if __name__ == '__main__':
   visualize_data(get_cleaned_processed_df())