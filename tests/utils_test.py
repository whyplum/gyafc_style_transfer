"""
Validate utils
"""

import numpy as np
from src.utils import slice_labels_to_batches


def test_slice_labels_to_batches(max_num_classes=5, max_samples_per_class=100, eps=.2, iterations=1000):
    """
    Validate slice_labels_to_batches
    :param max_num_classes: a random number of classes is drawn from {2..max_num_classes}
    :param max_samples_per_class: a random number samples per class is drawn from {10..max_samples_per_class}
    :param eps: maximum l1-diff in proportion of each class between batches
    :param iterations: number of times to repeat the test (with a different label vector)
    """

    for itr in range(iterations):

        # choose random number of classes
        num_classes = np.random.randint(2, max_num_classes)

        # for each class, choose random number of samples
        label_arr = []
        for c in range(num_classes):
            label_arr.append(c * np.ones(np.random.randint(10, max_samples_per_class)))

        # unify labels vectors
        labels = np.concatenate(label_arr)
        total_size = labels.shape[0]

        # choose random batch size, smaller than min(num samples per class)
        min_class_size = min([np.where(labels == c)[0].shape[0] for c in np.unique(labels)])
        num_batches = np.random.randint(1, min_class_size)

        # get batch slices
        batch_num_vector, shuffle_ind = slice_labels_to_batches(labels, num_batches)

        # reorder labels by slice shuffle
        shuffle_labels = np.take(labels, shuffle_ind)

        print("#######"
              "\nInit test.."
              "\nnum_classes %d"
              "\ntotal_size %d"
              "\nmin_class_size %d"
              "\nnum_batches %d" %
              (num_classes, total_size, min_class_size, num_batches))

        # make sure number of batches is correct
        assert np.unique(batch_num_vector).shape[0] == num_batches

        # collect label count for each batch
        batch_label_props = []
        for batch_num in np.unique(batch_num_vector):
            # get indices for batch (in shuffled space)
            batch_ind = np.where(batch_num_vector == batch_num)

            # extract labels of batch (in shuffled space)
            batch_labels = np.take(shuffle_labels, batch_ind)

            # keep label counts
            _, counts = np.unique(batch_labels, return_counts=True)

            # normalize counts to proportions
            prop = counts / np.sum(counts)
            batch_label_props.append(prop)

            print("batch", batch_num)
            print("counts", counts)

        # verify label proportion are ~constant in all batches
        for batch_prop in batch_label_props[:-1]:
            assert np.mean(np.abs((batch_label_props[0] - batch_prop))) <= eps

        # last batch is allowed to be offset by more, but must contain at least one sample from each class
        assert batch_label_props[0].shape == batch_label_props[-1].shape
