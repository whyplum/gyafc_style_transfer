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


def parse_arguments(args=None):
    p = argparse.ArgumentParser()

    p.add_argument('-d', '--debug', action="store_true", default=False)

    # logs
    p.add_argument('-sd', '--save-dir', type=str, default="save")
    p.add_argument('-ln', '--log-name', type=str, default="console.log")
    p.add_argument('-mn', '--metrics-name', type=str, default="metrics.pckl")
    p.add_argument('-cn', '--checkpoint-name', type=str, default="model.ckpt")

    # preprocessing
    p.add_argument('-ep', '--embedding-path', type=str, default="data/glove.42B.300d/filt.glove.42B.300d.txt")
    p.add_argument('-ed', '--embedding-size', type=int, default=300)
    p.add_argument('-dp', '--data-dir', type=str, default="data/E&M")

    # model
    # p.add_argument('-vs', '--vocab-size', type=int, default=1000)
    p.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    p.add_argument('-ru', '--num-rnn-units', type=int, default=16)
    p.add_argument('-cl', '--classifier-hidden-layers', nargs='+', type=int, default=[8, 4])
    p.add_argument('-mg', '--max-gradient-norm', type=float, default=1)

    # traning
    p.add_argument('-e', '--epoches', type=int, default=1000)
    p.add_argument('-b', '--batch-size', type=int, default=100)
    p.add_argument('-ss', '--save-step', type=int, default=5, help="Number of epoches between saves")
    p.add_argument('-le', '--limit-evals', type=int, default=500, help="if not -1, limit 'tune' and 'test' sets by number of samples")
    p.add_argument('-r', '--restore', action="store_true", default=False)

    args = p.parse_args(args)
    return args


def shuffle_and_batch(data, data_type, batch_size):

    logging.info("Slicing %s data to batches with consistent class proportion..." % data_type)
    batch_num_vector, shuffle_ind = slice_labels_to_batches(
        data[data_type]["labels"],
        ceil(data[data_type]["labels"].shape[0] / batch_size))

    _, counts = np.unique(batch_num_vector, return_counts=True)
    logging.info("%s batch sizes: %s" % (data_type, counts))

    # reorder data by slice shuffle
    logging.info("Shuffling %s data..." % data_type)
    shuffle = lambda x, ax: np.take(x, shuffle_ind, axis=ax)

    X, X_len, Y = (shuffle(x, ax) for x, ax in (
        (data[data_type]["sentences"], 1),
        (data[data_type]["sentence_lengths"], 0),
        (data[data_type]["labels"], 0)))

    return X, X_len, Y, batch_num_vector


def extract_batch(X, X_len, Y, batch_num_vector, batch_num):

    # get indices for batch (in shuffled space)
    batch_ind = np.where(batch_num_vector == batch_num)[0]

    # extract data of batch (in shuffled space)
    batch_slice = lambda x, ax: np.take(x, batch_ind, axis=ax)
    batch_X, batch_X_len, batch_Y = (batch_slice(x, ax) for x, ax in ((X, 1), (X_len, 0), (Y, 0)))

    return batch_X, batch_X_len, batch_Y


def eval_step(sess, model, saver,
              ghistory, gstep,
              embedder, X, X_len, Y, batch_num_vector,
              data_type, debug=False, update=False, save=False):

    logging.info("Evaluating %s set.." % data_type)

    for batch_num in np.unique(batch_num_vector):

        # extract batch data
        batch_X, batch_X_len, batch_Y = extract_batch(X, X_len, Y, batch_num_vector, batch_num)

        # run batch
        metrics = model.step(sess, embedder, batch_X, batch_X_len, batch_Y, update)

        # save metrics
        for k, v in metrics.items():
            ghistory.loc[ghistory.shape[0]] = (data_type, gstep.eval(), batch_num, k, v)
            if batch_num == 0:
                logging.info("%s | Epoch %d | Batch %d | Metric %s | %.5f" % (data_type, gstep.eval(), batch_num, k, v))

        # allow early stopping
        if debug and batch_num >= 1:
            logging.info("Debug mode: breaking after 2 batches..")
            break

    # save final model and metrics
    if save:
        ghistory.to_pickle(os.path.join(args.save_dir, args.metrics_name))
        saver.save(sess, os.path.join(args.save_dir, args.checkpoint_name), gstep)


def train(args):

    # create necessary dirs if not exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # init logger
    configure_logger(args.save_dir, args.log_name)
    logging.info("Got arguments: %s" % args.__dict__)

    logging.info("Getting Dataset...")
    data = preprocess_dataset(get_datasets(args.data_dir, "formal"), get_datasets(args.data_dir, "informal"),
                             args.embedding_path, args.embedding_size, args.limit_evals)

    # extract train data
    X, X_len, Y = (data["train"][v] for v in ("sentences", "sentence_lengths", "labels"))
    logging.info("Found train/tune/test inputs of dims %s/%s/%s, embedder of dims %s"
                 % (X.shape, data["tune"]["sentences"].shape, data["test"]["sentences"].shape, data["embedding"].shape))

    if args.debug:
        logging.info("Entering debug mode: training for a maximum of 2 batches")

    # shuffle and batch
    trn_X, trn_X_len, trn_Y, trn_batches = shuffle_and_batch(data, "train", args.batch_size)
    tun_X, tun_X_len, tun_Y, tun_batches = shuffle_and_batch(data, "tune", args.batch_size)
    tst_X, tst_X_len, tst_Y, tst_batches = shuffle_and_batch(data, "test", args.batch_size)

    logging.info("Building Model and Saver...")

    # build model
    model = GruClassifier(deep_dict_defaults({
        'vocab_size': data["embedding"].shape[0], 'embedding_size': data["embedding"].shape[1],
        'max_time': trn_X.shape[0], }, args.__dict__), mode='train')
    logging.info("Created model %s" % model)

    # keep a global step counter, and build saver
    gstep = tf.Variable(-1, name="global_step", trainable=False, dtype=tf.int32)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.epoches)

    logging.info("Starting TensorFlow session...")
    with tf.Session() as sess:

        # restore or init variables
        if args.restore:
            try:
                ghistory = pd.read_pickle(os.path.join(args.save_dir, args.metrics_name))
                saver.restore(sess, tf.train.latest_checkpoint(args.save_dir))
            except (ValueError, FileNotFoundError):
                logging.error("Failed to restore variables or metrics.")
                raise
        else:
            ghistory = pd.DataFrame(columns=("data", "epoch", "batch", "metric", "value"))
            sess.run(tf.global_variables_initializer())

        try:
            for epoch_num in range(gstep.eval() + 1, args.epoches):
                _ = sess.run(tf.assign(gstep, epoch_num))

                # evaluate train set
                eval_step(sess, model, saver,
                          ghistory, gstep,
                          data["embedding"], trn_X, trn_X_len, trn_Y, trn_batches,
                          "train", args.debug, update=False)

                # evaluate tune dataset
                if epoch_num % args.save_step == 0:
                    eval_step(sess, model, saver,
                              ghistory, gstep,
                              data["embedding"], tun_X, tun_X_len, tun_Y, tun_batches,
                              "tune", args.debug, save=True)
        except KeyboardInterrupt:
            pass

        # evaluate test dataset
        eval_step(sess, model, saver,
                  ghistory, gstep,
                  data["embedding"], tst_X, tst_X_len, tst_Y, tst_batches,
                  "test", args.debug, save=True)

    logging.info('#' * 10 + "Training is COMPLETE!" + '#' * 10)


if __name__ == "__main__":
    args = parse_arguments()
    train(args)
