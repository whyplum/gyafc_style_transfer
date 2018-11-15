import os
import logging
import argparse

import tensorflow as tf
import numpy as np
import pandas as pd
from math import ceil

from src.models import GruClassifier
from src.data_preprocess import preprocess_dataset, get_datasets
from src.utils import configure_logger, deep_dict_defaults, slice_labels_to_batches

def parse_arguments():
    p = argparse.ArgumentParser()

    p.add_argument('-d', '--debug', action="store_true", default=False)

    # logs
    p.add_argument('-sd', '--save-dir', type=str, default="save")
    p.add_argument('-ln', '--log-name', type=str, default="console.log")

    # preprocessing
    p.add_argument('-ep', '--embedding-path', type=str, default="data/glove.42B.300d/filt.glove.42B.300d.txt")
    p.add_argument('-ed', '--embedding-size', type=int, default=300)
    p.add_argument('-dp', '--data-dir', type=str, default="data/E&M")

    # model
    # p.add_argument('-vs', '--vocab-size', type=int, default=1000)
    # p.add_argument('-mt', '--max-time', type=int, default=25)
    p.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    p.add_argument('-ru', '--num-rnn-units', type=int, default=16)
    p.add_argument('-cl', '--classifier-hidden-layers', nargs='+', type=int, default=[8, 4])
    p.add_argument('-mg', '--max-gradient-norm', type=float, default=1)

    # traning
    p.add_argument('-e', '--epoches', type=int, default=1000)
    p.add_argument('-b', '--batch-size', type=int, default=100)
    p.add_argument('-ss', '--save-step', type=int, default=5, help="Number of epoches between saves")
    p.add_argument('-le', '--limit-evals', type=int, default=500, help="if not -1, limit 'tune' and 'test' sets by number of samples")

    # p.add_argument('--display_step', type=int, default=10)
    # p.add_argument('--input_size', type=int, default=1)
    # p.add_argument('--time_steps', type=int, default=100)
    # p.add_argument('--hidden_features', type=int, default=256)
    # p.add_argument('--output_classes', type=int, default=127)
    # p.add_argument('--num_layers', type=int, default=3)

    args = p.parse_args()
    return args

def train(args):
    # create necessary dirs if not exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # init logger
    configure_logger(args.save_dir, args.log_name)
    logger = logging.getLogger(__name__)

    logger.info("Getting Dataset...")
    data = preprocess_dataset(get_datasets(args.data_dir, "formal"), get_datasets(args.data_dir, "informal"),
                             args.embedding_path, args.embedding_size, args.limit_evals)

    # split data results to train / tune / test
    X, X_len, Y = (data["train"][v] for v in ("sentences", "sentence_lengths", "labels"))
    logging.info("Found train/tune/test inputs of dims %s/%s/%s, embedder of dims %s"
                 % (X.shape, data["tune"]["sentences"].shape, data["test"]["sentences"].shape, data["embedding"].shape))

    if args.debug:
        logging.info("Entering debug mode: training for a maximum of 2 batches")

    logging.info("Slicing data to batches with consistent class proportion...")
    batch_num_vector, shuffle_ind = slice_labels_to_batches(Y, ceil(Y.shape[0] / args.batch_size))
    _, counts = np.unique(batch_num_vector, return_counts=True)
    logging.info("Batch sizes: %s" % counts)

    # reorder data by slice shuffle
    logging.info("Shuffling data...")
    shuffle = lambda x, ax: np.take(x, shuffle_ind, axis=ax)
    shuff_X, shuff_X_len, shuff_Y = (shuffle(x, ax) for x, ax in ((X, 1), (X_len, 0), (Y, 0)))

    logger.info("Building Model and Saver...")
    model = GruClassifier(deep_dict_defaults({
        'vocab_size': data["embedding"].shape[0], 'embedding_size': data["embedding"].shape[1],
        'max_time': shuff_X.shape[0], }, args.__dict__), mode='train')
    saver = tf.train.Saver(tf.global_variables())
    logging.info("Created model %s" % model)

    logger.info("Starting TensorFlow session...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        metrics = ("loss", "accuracy")
        global_history = pd.DataFrame(columns=("data", "epoch", "metric", "value"))
        for epoch_num in range(args.epoches):
            batch_history = {k: [] for k in metrics}
            for batch_num in np.unique(batch_num_vector):
                # get indices for batch (in shuffled space)
                batch_ind = np.where(batch_num_vector == batch_num)[0]

                # extract data of batch (in shuffled space)
                batch_slice = lambda x, ax: np.take(x, batch_ind, axis=ax)
                batch_X, batch_X_len, batch_Y = (batch_slice(x, ax) for x, ax in
                                                 ((shuff_X, 1), (shuff_X_len, 0), (shuff_Y, 0)))

                # run batch
                feed = {model._embedder: data["embedding"], model._input_lengths: batch_X_len,
                        model._inputs: batch_X, model._labels: batch_Y}
                result = sess \
                    .run(feed_dict=feed,
                         fetches=[model._update_step, model._loss, model._accuracy])

                # save loss and accuracy
                for i in range(1, len(metrics) + 1):
                    batch_history[metrics[i - 1]].append(result[i].flatten())

                if args.debug and batch_num >= 1:
                    logging.info("Debug mode: breaking after 2 batches..")
                    break

            # save loss and accuracy
            for i in range(1, len(metrics) + 1):
                # calculate epoch mean metrics
                metric_epoch_mean = np.mean(np.concatenate(batch_history[metrics[i - 1]]))

                # save and log it
                global_history.loc[global_history.shape[0]] = ("train", epoch_num, metrics[i - 1], metric_epoch_mean)
                logger.info("Train | Epoch %d | Metric %s | %.5f" % (epoch_num, metrics[i - 1], metric_epoch_mean))

            if epoch_num % args.save_step == 0:
                # save model and metrics checkpoint
                global_history.to_pickle(os.path.join(args.save_dir, "metrics.pckl"))
                saver.save(sess, os.path.join(args.save_dir, "e%d.ckpt" % epoch_num))

                # evaluate tune dataset
                logger.info("Epoch %d | Evaluating tune set.." % epoch_num)
                feed = {model._embedder: data["embedding"], model._input_lengths: data["tune"]["sentence_lengths"],
                        model._inputs: data["tune"]["sentences"], model._labels: data["tune"]["labels"]}
                result = sess \
                    .run(feed_dict=feed,
                         fetches=[model._loss, model._accuracy])

                # save tune evaluation metrics
                for i in range(len(metrics)):
                    metric_eval_mean = np.mean(result[i].flatten())
                    global_history.loc[global_history.shape[0]] = ("tune", epoch_num, metrics[i], metric_eval_mean)
                    logger.info("Tune | Epoch %d | Metric %s | %.5f" % (epoch_num, metrics[i], metric_eval_mean))

        # evaluate tune dataset
        logger.info("Evaluating test set..")
        feed = {model._embedder: data["embedding"], model._input_lengths: data["test"]["sentence_lengths"],
                        model._inputs: data["test"]["sentences"], model._labels: data["test"]["labels"]}
        result = sess \
            .run(feed_dict=feed,
                 fetches=[model._loss, model._accuracy])

        # save test evaluation metrics
        for i in range(len(metrics)):
            metric_eval_mean = np.mean(result[i].flatten())
            global_history.loc[global_history.shape[0]] = ("test", None, metrics[i], metric_eval_mean)
            logger.info("Test | Metric %s | %.5f" % (metrics[i], metric_eval_mean))

        # save final model and metrics
        saver.save(sess, os.path.join(args.save_dir, "model.ckpt"))
        global_history.to_pickle(os.path.join(args.save_dir, "metrics.pckl"))

    logger.info('#' * 10 + "Training is COMPLETE!" + '#' * 10)


if __name__ == "__main__":
    args = parse_arguments()
    train(args)