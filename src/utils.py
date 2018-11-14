import decorator

import tensorflow as tf
import numpy as np


def deep_dict_defaults(current, defaults):
    """
    Fill in default values in a dictionary.
    For each key in default that doesn't exist in current, fill in the value from defaults.
    If one of these values is a dict, recursivly call the function on that dict.
    :param current: dict
    :param defaults: dict
    :return: filled in dict
    """

    d = defaults.copy()
    d.update(current)

    for k,v in defaults.items():
        if isinstance(v, dict):
            cur_v = current.get(k, None)
            d[k] = deep_dict_defaults(cur_v if isinstance(cur_v, dict) else dict(), v)

    return d


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])


def clean_tf_graph(func):
    def wrapper(func, *args, **kwargs):
        # clean tf computational graph
        tf.reset_default_graph()

        return func(*args, **kwargs)
    return decorator.decorator(wrapper, func)


def configure_logger(log_dir=None, file_name=None, level="DEBUG"):
    """
    In case log_path and file_name are not None and form a valid path, log will be saved to log_path/file_name/<num>.txt
    """
    import logging
    from os.path import join

    logFormatter = logging.Formatter("[%(asctime)s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(level)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    new_session_msg = '#' * 10 + " New session " + '#' * 10

    if log_dir is not None and file_name is not None:
        path = join(log_dir, file_name)

        try:
            fileHandler = logging.FileHandler(path, mode='a')
            fileHandler.setFormatter(logFormatter)
            rootLogger.addHandler(fileHandler)
        except (FileNotFoundError, OSError):
            rootLogger.info(new_session_msg)
            rootLogger.error("Couldn't create/append to log file at %s" % path)
        else:
            rootLogger.info(new_session_msg)
            rootLogger.info("Successfully connected to log file at %s" % path)
    else:
        rootLogger.info(new_session_msg)
        rootLogger.info("Logging to stdout only")


def slice_to_batches_by_batch_size(total_size, batch_size):
    """
    Return a list of batch indices
    :param total_size: number of total records
    :param batch_size: number of records in a single batch
    :return: list of (batch_num, (start_ind, end_ind))
    """

    # naive slicing
    slices = list(range(0, total_size, batch_size))

    # append last if not exists
    if slices[-1] != total_size - 1:
        slices.append(total_size - 1)

    # last slice is inclusive
    slices[-1] = total_size

    # refactor to (batch_num, (start_ind, end_ind))
    refactored = list(enumerate(zip(slices[:-1], slices[1:])))

    return refactored


def slice_to_batches_by_num_batches(total_size, num_batches):
    """
    Return a list of batch indices
    :param total_size: number of total records
    :param num_batches: total number of batches
    :return: list of (batch_num, (start_ind, end_ind))
    """

    batch_size = total_size // num_batches
    reminder = total_size - (num_batches * (total_size // num_batches))

    slices = [(-1, (-1, -1)),]
    for batch in range(num_batches):
        start_ind = slices[-1][1][1]  # start index of cur slice == end index of last slice
        end_ind = start_ind + batch_size

        if reminder > 0:
            end_ind += 1
            reminder -= 1

        slices.append((batch, (start_ind, end_ind)))

    return slices[1:]


def slice_labels_to_batches(labels, num_batches):
    """
    Return a list of batch indices
    :param labels: flat numpy array with labels
    :param num_batches: total number of batches
    :return:
      1. list of (batch_num, (start_ind, end_ind)),
           such that each batch has the same number of records from each label up to 1 sample
      2. reshuffling index which must be applied before using batch slices
    """

    # labels are assumed to be flat
    labels = labels.flatten()

    total_size = labels.shape[0]

    # shuffle labels
    shuffle_ind = np.arange(total_size)
    np.random.shuffle(shuffle_ind)
    labels = np.take(labels, shuffle_ind)

    batch_num_vector = -np.ones(total_size)

    # split by class label
    for c in np.unique(labels):

        # we want to keep potion of each class in every batch. Hence:
        #  class_size // class_batch_size == total_size // batch_size
        class_ind = np.where(labels == c)[0]
        class_size = class_ind.shape[0]

        # split
        class_batch_slices = slice_to_batches_by_num_batches(class_size, num_batches)

        for batch_num, (bs, be) in class_batch_slices:
            np.put(batch_num_vector, np.take(class_ind, range(bs, be)), batch_num)

    return batch_num_vector, shuffle_ind


def batch_embedding_lookup(inputs, embedder, final_size):
    """
    Handle None dimension by flattening the vector before lookup
    :param inputs: 2-dim tensor
    :param embedder: 2-dim tensor
    :param final_size: list of len 3, where one of values is -1 (None)
    :return: embedded tensor in sim final_size
    """
    emb_flat = tf.nn.embedding_lookup(
        embedder, tf.reshape(inputs, [-1]))
    emb_inp = tf.reshape(emb_flat, final_size)
    return emb_inp


def optimize(optimizer, loss, parameters, clip_norm):
    """
    Use optimizer to minimize loss by optimizing parameters, while using gradient norm clipping
    :param optimizer: tf optimizer
    :param loss: tf tensor to minimize
    :param parameters: collection of tf tensors to optimize over
    :param clip_norm: float to clip gradients by
    :return: update_step tf operation
    """

    # Calculate and clip gradients
    gradients = tf.gradients(loss, parameters)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)

    # Optimization
    update_step = optimizer.apply_gradients(zip(clipped_gradients, parameters))

    return update_step
