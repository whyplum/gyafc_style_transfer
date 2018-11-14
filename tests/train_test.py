"""
Validate model's training process by training on mock data
"""

import tensorflow as tf
import numpy as np

from src.models import Encoder, GruClassifier
from src.utils import clean_tf_graph


@clean_tf_graph
def test_gru_classifier(decrease_perc=0.8):
    """
    Train GruClassifier
    Assert loss is strictly decreasing at least decrease_perc of time
    :param decrease_perc: real in range [0,1]
    """

    hparams = {
        'vocab_size': 1000,
        'embedding_size': 10,
        'max_time': 5,
        'num_rnn_units': 16,
        'classifier_hidden_layers': [8, 4],
        'max_gradient_norm': 1,
        'learning_rate': 1e-3,
        'epoch': 1000,
        'mock_size': 200  # needs to be an Even number, for later split
    }

    print("Generating mock data")

    # use model.test() to generate mock data for two classes; and clean tf graph
    enc = Encoder(hparams)
    data = enc.test(hparams["mock_size"])
    tf.reset_default_graph()

    # label and shuffle classes
    shuffle_ind = np.arange(hparams["mock_size"])
    np.random.shuffle(shuffle_ind)
    X = np.take(data["input_sentences"], shuffle_ind, axis=1)
    X_len = data["input_lengths"][shuffle_ind]
    Y = np.concatenate((np.ones(hparams["mock_size"] // 2), -np.ones(hparams["mock_size"] // 2)))[shuffle_ind]

    print("Data generated: X %s X_len %s Y %s" % (X.shape, X_len.shape, Y.shape))

    # build model
    print("Building model")
    model = GruClassifier(hparams)

    # metrics to log
    metrics = ("loss", "accuracy")

    print("Initializing TF session")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        history = {k: [] for k in metrics}
        for epoch_num in range(hparams["epoch"]):

            feed = {model._embedder: data["embedder"], model._input_lengths: X_len,
                    model._inputs: X, model._labels: Y}

            result = sess.run([model._update_step, model._loss, model._accuracy],
                              feed_dict=feed)

            # save loss and accuracy
            for i in range(1, len(metrics) + 1):
                agg = np.mean(result[i].flatten())
                history[metrics[i - 1]].append(agg)

        # assert losses are strictly decreasing at least decrease_perc of time
        assert (np.mean(np.diff(history["loss"]) < 0) >= decrease_perc)
