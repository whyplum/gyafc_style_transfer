import abc

import tensorflow as tf
import numpy as np

from src.utils import deep_dict_defaults, batch_embedding_lookup, optimize

class Model(object):
    """
    An abstract Neural Network model
    """

    DEFAULT_HPARAMS = {}

    def __init__(self, args, scope, build):
        """
        :param args: argument dict or object with __dict__ attribute. Overrides defaults from class.DEFAULT_HPARAMS
        :param scope: tf scope
        :param build: should the model be built on init
        """

        if isinstance(args, dict):
            inp_hparams = args
        else:
            if hasattr(args, '__dict__'):
                inp_hparams = vars(args)
            else:
                inp_hparams = {}

        # fill hyper_params with default, where non were provided
        self.hparams = deep_dict_defaults(inp_hparams, type(self).DEFAULT_HPARAMS)

        self._scope = scope

    @abc.abstractmethod
    def build(self):
        """
        Build tf graph
        """
        pass

    @abc.abstractmethod
    def test(self, mock_size=100, **kwargs):
        """
        Test validity of tf graph
        Use input data or generate mock data if None is provided
        :return: dict of model, data and results
        :param mock_size: number of mock records to generate, in case input is None
        """
        return {}

    def __str__(self):
        return "[%s %s (%s)]" % (self.__class__.__name__, self._scope, self.hparams)


class Encoder(Model):
    DEFAULT_HPARAMS = deep_dict_defaults({
        'src_vocab_size': 1000,
        'embedding_size': 10,
        'max_time': 5,
        'num_units': 16,
    }, Model.DEFAULT_HPARAMS)

    def __init__(self, args=None, scope='encoder', build=True):
        super(Encoder, self).__init__(args, scope, build)

        # input placeholders
        with tf.name_scope(self._scope):
            self._inputs = tf.placeholder(tf.int32, [self.hparams['max_time'], None], name="inputs")
            self._input_lengths = tf.placeholder(tf.int32, [None,], name="input_lengths")
            self._embedder = tf.placeholder(tf.float32, [self.hparams['src_vocab_size'], self.hparams['embedding_size']], name="embedder")

        # outputs to be init in build
        self._emb_inp = None
        self._outputs = None
        self._hidden_states = None

        # build the model
        if build: self.build()

    def build(self):

        with tf.name_scope(self._scope):

            # Look up embedding
            #   encoder_emb_inp: [max_time, batch_size, embedding_size]
            self._emb_inp = batch_embedding_lookup(self._inputs, self._embedder,
                                                   [self.hparams['max_time'], -1, self.hparams['embedding_size']])

            # Dynamic RNN
            #   encoder_outputs: [max_time, batch_size, num_units]
            #   encoder_state:   [batch_size, num_units]
            encoder_cell = tf.nn.rnn_cell.GRUCell(self.hparams['num_units'], name='basic_gru_cell')
            self._outputs, self._hidden_states = tf.nn.dynamic_rnn(
                encoder_cell, inputs=self._emb_inp, dtype=tf.float32,
                sequence_length=self._input_lengths, time_major=True)

    def test(self, mock_size=100, embedder=None, input_sentences=None, input_lengths=None):
        """
        :param mock_size: number of mock records to generate, in case input is None
        :param embedder: embedding vectors in range [-1, 1] of dim (src_vocab_size, embedding_size)
        :param input_sentences: word indices in {0..src_vocab_size} of dim (max_time, batch_size)
        :param input_lengths: sentence lengths in {1..(max_time+1)} of dim (batch_size). Note that each sentence should end with an END token
        """

        # Randomize embedding and sentences, if they aren't provided
        if embedder is None:
            embedder = np.random.uniform(low=-1, high=1,
                size=[self.hparams['src_vocab_size'], self.hparams['embedding_size']])
        if input_sentences is None:
            input_sentences = np.random.randint(low=0, high=self.hparams['src_vocab_size'],
                size=[self.hparams['max_time'], mock_size])
        if input_lengths is None:
            input_lengths = np.random.randint(low=1, high=self.hparams['max_time'] + 1,
                size=[mock_size,])

            # max sure at least one sentence has max length
            input_lengths[-1] = self.hparams['max_time']

        feed = {
            self._inputs: input_sentences,
            self._input_lengths: input_lengths,
            self._embedder: embedder}

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            result = sess.run({"outputs": self._outputs, "hidden_states": self._hidden_states}, feed_dict=feed)
            outputs, hidden_state = result["outputs"], result['hidden_states']

            assert outputs.shape == \
                   (self.hparams['max_time'], mock_size, self.hparams['num_units'])

            assert hidden_state.shape == \
                   (mock_size, self.hparams['num_units'])

            return {'embedder': embedder,
                    'input_sentences': input_sentences,
                    'input_lengths': input_lengths,
                    'outputs': outputs,
                    'hidden_states': hidden_state}


class Perceptron(Model):
    DEFAULT_HPARAMS = deep_dict_defaults({
        'num_inputs': 16,
        'num_hidden_layers': 2,
        'num_labels': 2,
        'num_units': [8, 4],
    }, Model.DEFAULT_HPARAMS)

    def __init__(self, args=None, scope='perceptron', build=True):
        super(Perceptron, self).__init__(args, scope, build)

        # input placeholders
        with tf.name_scope(self._scope):
            self._inputs = tf.placeholder(tf.float32, [None, self.hparams['num_inputs']], name="inputs")

        # outputs to be init in build
        self._logits = None
        self._predictions = None

        # build the model
        if build: self.build()

    def build(self):

        with tf.name_scope(self._scope):

            # Multilayer perceptron
            cur = self._inputs
            for i in range(self.hparams['num_hidden_layers']):
                cur = tf.contrib.layers.fully_connected(
                    inputs=cur,
                    num_outputs=self.hparams['num_units'][i])

            # last layer is linear
            self._logits = tf.contrib.layers.fully_connected(
                inputs=cur,
                num_outputs=self.hparams['num_labels'],
                activation_fn=None)

            self._predictions = tf.contrib.layers.softmax(self._logits)

    def test(self, mock_size=100, inputs=None):
        """
        :param mock_size: number of mock records to generate, in case input is None
        :param inputs: inputs in range [-1, 1] of dim (batch_size, num_inputs)
        """

        # Randomize inputs, if they aren't provided
        if inputs is None:
            inputs = np.random.uniform(low=-1, high=1,
                size=[mock_size, self.hparams['num_inputs']])

        feed = {self._inputs: inputs,}

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            result = sess.run({"logits": self._logits, "predictions": self._predictions}, feed_dict=feed)
            logits, predictions = result["logits"], result['predictions']

            assert logits.shape == predictions.shape == \
                   (mock_size, self.hparams['num_labels'])

            return {'inputs': inputs,
                    'logits': logits,
                    'predictions': predictions}


class GruClassifier(Model):
    """
        RNN based classifier . using pretrained embedding, gru layer, fully connected layer
    """
    DEFAULT_HPARAMS = deep_dict_defaults({
            'vocab_size': 1000,
            'embedding_size': 10,
            'max_time': 25,
            'num_rnn_units': 16,
            'classifier_hidden_layers': [8, 4],
            'max_gradient_norm': 1,
            'learning_rate': 0.001,
            'num_labels': 2
        }, Model.DEFAULT_HPARAMS)

    def __init__(self, args=None, scope='gru_classifier', build=True, mode='train'):
        """
        :param mode: train or eval
        """
        super(GruClassifier, self).__init__(args, scope, build)

        # assert mode is valid
        assert mode.lower() in ['train', 'eval']
        self.mode = mode.lower()

        # input placeholders
        with tf.name_scope(self._scope):
            self._inputs = tf.placeholder(tf.int32, [self.hparams['max_time'], None], name="inputs")
            self._input_lengths = tf.placeholder(tf.int32, [None,], name="input_lengths")
            self._embedder = tf.placeholder(tf.float32, [self.hparams['vocab_size'], self.hparams['embedding_size']], name="embedder")
            self._labels = tf.placeholder(tf.int32, [None,], name="labels")

        # outputs to be init in build
        self._encoder = None
        self._loss = None
        self._update_step = None
        self._prediction = None

        # build the model
        if build: self.build()

    @property
    def component_hparams(self):
        return {
            'encoder': {
                'src_vocab_size': self.hparams['vocab_size'],
                'embedding_size': self.hparams['embedding_size'],
                'max_time': self.hparams['max_time'],
                'num_units': self.hparams['num_rnn_units'],
            },
            'perceptron': {
                'num_inputs': self.hparams['num_rnn_units'],
                'num_hidden_layers': len(self.hparams['classifier_hidden_layers']),
                'num_labels': 2,
                'num_units': self.hparams['classifier_hidden_layers'],
            }
        }

    def build(self):
        # input -> encoded info vector
        self._encoder = Encoder(self.component_hparams['encoder'], build=False)
        self._encoder._inputs = self._inputs
        self._encoder._input_lengths = self._input_lengths
        self._encoder._embedder = self._embedder
        self._encoder.build()

        # encoded info vector -> class label
        self._perceptron = Perceptron(self.component_hparams['perceptron'], build=False)
        self._perceptron._inputs = self._encoder._hidden_states
        self._perceptron.build()

        # combined loss
        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.one_hot(indices=self._labels, depth=2), logits=self._perceptron ._logits))

        tf.summary.scalar('loss', self._loss)

        # Optimization
        optimizer = tf.train.AdamOptimizer(self.hparams['learning_rate'])
        self._update_step = optimize(optimizer, self._loss, tf.trainable_variables(), self.hparams['max_gradient_norm'])

        # Calculate perceptron accuracy
        self._prediction = self._perceptron._predictions

        perc_pred_choice = tf.cast(tf.argmax(self._perceptron._predictions, 1), tf.int32)
        self._accuracy = tf.reduce_mean(tf.cast(tf.equal(perc_pred_choice, self._labels), tf.float32))
        tf.summary.scalar('accuracy', self._accuracy)

    def step(self, sess, embedder, inputs, input_lengts, labels, update=True):

        feed = {
            self._embedder: embedder,
            self._inputs: inputs,
            self._input_lengths: input_lengts,
            self._labels: labels}

        fetches = {
            "loss": self._loss,
            "accuracy": self._accuracy
        }

        if update:
            fetches["update_step"] = self._update_step

        results = sess.run(feed_dict=feed, fetches=fetches)

        if update:
            results.pop("update_step")

        # average metrics
        metrics = {k: np.mean(v.flatten()) for k,v in results.items()}

        return metrics

    def test(self, mock_size=100, embedder=None, input_sentences=None, input_lengths=None, sentence_labels=None):
        """
        :param mock_size: number of mock records to generate, in case input is None
        :param embedder: embedding vectors in range [-1, 1] of dim (src_vocab_size, embedding_size)
        :param input_sentences: word indices in {0..src_vocab_size} of dim (max_time, batch_size)
        :param input_lengths: sentence lengths in {1..(max_time+1)} of dim (batch_size). Note that each sentence should end with an END token
        :param sentence_labels: sentence labels in {0, 1} of dim (batch_size)
        """

        # Randomize embedding and sentences, if they aren't provided
        if embedder is None:
            embedder = np.random.uniform(low=-1, high=1,
                size=[self.hparams['vocab_size'], self.hparams['embedding_size']])
        if input_sentences is None:
            input_sentences = np.random.randint(low=0, high=self.hparams['vocab_size'],
                size=[self.hparams['max_time'], mock_size])
        if input_lengths is None:
            input_lengths = np.random.randint(low=1, high=self.hparams['max_time'] + 1,
                size=[mock_size,])
            # make sure at least one sentence has max length
            input_lengths[-1] = self.hparams['max_time']
        if sentence_labels is None:
            sentence_labels = np.random.randint(low=0, high=2,
                size=[mock_size,])
            sentence_labels[sentence_labels == 0] = -1  # labels should be in {+1, -1}

        feed = {
            self._inputs: input_sentences,
            self._input_lengths: input_lengths,
            self._embedder: embedder,
            self._labels: sentence_labels
        }

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            result = sess.run({"loss": self._loss , "update_step": self._update_step}, feed_dict=feed)
            loss, update_step = result["loss"], result['update_step']

            assert isinstance(loss, (np.float, np.float16, np.float32, np.float64))

            return {'embedder': embedder,
                    'input_sentences': input_sentences,
                    'input_lengths': input_lengths,
                    'sentence_labels': sentence_labels,
                    'loss': loss,
                    'update_step': update_step}