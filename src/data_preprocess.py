"""
Handle E&M and GloVe data
=========================

find the MAX_WORDS most frequent words

create a dictionary of word to integer (with UNK)

replace all infrequent words with UNK + add START and END tokens
find the longest sentence
create a matrix MAX_SENT_LEN * dataset_size s.t each entry has the index of the word
vector of all sentences length - for any sentence count the number of words in the sentence
lables vector

load the embeddings GloVe - return a matrix s.t the ith row is the embedding of the ith word in the vocabulary
"""

import re
from os.path import join
import numpy as np

# set numpy random seed
SEED = 7
np.random.seed(SEED)

# global peprocessing parameters
MAX_WORDS = 10000
EPS = 1.0
UNK = "<unk>"
START = "<s>"
END = "<s/>"
PAD = "<p>"
MAX_SENT_LEN = 25
MAX_SENT_LEN_W_TOKENS = MAX_SENT_LEN + 2
FORMAL = 0
INFORMAL = 1
DATASET_TYPES = ["train", "tune", "test"]


def preprocess_dataset(formalsets, informalsets, embedding_file, embedding_dims,
                       limit_evals=-1, inttype=np.int32):
    """
    Preprocess the dataset for embedding lookup
    :param formalsets: dict(with keys DATASET_TYPES) of files of formal sentences
    :param informalsets: dict(with keys DATASET_TYPES) of files of informal sentences
    :param limit_evals: if not -1, limit 'tune' and 'test' sets to limit_evals samples
    :param inttype: the int type to use for the output numpy vectors
    :return: dict of:
        word2idx and idx2word dictionaries
        for each key in DATASET_TYPES, dict of:
            sentences dataset, sentence lengths
            sentence labels
    """

    assert isinstance(formalsets, dict)
    assert isinstance(informalsets, dict)
    assert all([k in formalsets for k in DATASET_TYPES])
    assert all([k in informalsets for k in DATASET_TYPES])

    # create vocabulary
    full_vocab = create_vocabulary(list(formalsets.values()) + list(informalsets.values()))

    # create embedding [full_glove_words == (Glove vocab) intersect (full vocab from dataset)]
    full_embedding, full_glove_vocab = get_glove_word_embedding(full_vocab, embedding_file, embedding_dims)

    # filter top MAX_WORDS
    top_vocab = full_glove_vocab[:MAX_WORDS]
    top_embedding = full_embedding[:MAX_WORDS, :]

    # create a word2idx map for full_vocab
    word2idx = get_word2idx(full_vocab, top_vocab)

    res = {"word2idx": word2idx, "embedding": top_embedding}
    for data_type in DATASET_TYPES:

        # load and preprocess sentences
        formal_docs = load_all_data_sets(formalsets[data_type])
        informal_docs = load_all_data_sets(informalsets[data_type])

        if data_type in ("tune", "test") and limit_evals > 0:
            formal_docs = formal_docs[:limit_evals // 2]
            informal_docs = informal_docs[:limit_evals // 2]

        docs = formal_docs + informal_docs

        # generate labels
        formal_labels = np.array([FORMAL] * len(formal_docs))
        informal_labels = np.array([INFORMAL] * len(informal_docs))
        labels = np.concatenate((formal_labels, informal_labels), axis=0).astype(inttype)

        # calculate sentence length
        sents_length = get_sents_length_vector(docs).astype(inttype)

        # sentences to matrix of indices
        dataset = sents_to_integers(docs, word2idx).astype(inttype).T

        res[data_type] = {"labels": labels, "sentence_lengths": sents_length, "sentences": dataset}

    return res


def preprocess_line(line):
    """
    Preprocess a line into a clean list of words.
      change all letters to lowercase
      strip whitespaces
      create a space between a word and the punctuation following it
      remove double usage of ".", "?", "!", ","
      leave only these characters (a-z, A-Z, ".", "?", "!", ",", whitespace)
    :param line: string which represents a file line
    :return: a list of cleaned words
    """

    new_line = line.lower().strip()

    # replacing everything with nothing except (a-z, A-Z, ".", "?", "!", ",", whitespace)
    new_line = re.sub(r"[^a-zA-Z?.!,¿\s]+", "", new_line)

    # remove double usage of ".", "?", "!", ","
    for c in ["\.", "\,", "\!", "\?"]:
        new_line = re.sub("(%s+)" % c, "?", new_line)

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    new_line = re.sub(r"([?.!,¿])", r" \1 ", new_line)
    new_line = re.sub(r'[" "]+', " ", new_line)

    new_line = new_line.rstrip().strip()
    words = new_line.split(' ')

    return words


def create_vocabulary(files):
    """
    Returns a set of the vocabulary, with frequency, ordered by frequency and word
    I.e. [(wrd, freq), ..]
    """
    # Read all files and get all tokens, count frequency
    cntr = {}
    for file in files:
        with open(file, 'r', encoding='UTF-8') as fin:
            for line in fin:
                for wrd in preprocess_line(line):
                    cntr[wrd] = cntr.get(wrd, 0) + 1

    # get most frequent words
    # sort by cnt, key. both are needed to avoid inconsistency in function call.
    sorted_by_word = sorted(list(cntr.items()), key=lambda x: x[0])
    sorted_by_freq_word = sorted(sorted_by_word, key=lambda x: x[1], reverse=True)

    # add tokens as most frequent words
    words_by_freq_word = [START, END, UNK, PAD] + [wrd for wrd, freq in sorted_by_freq_word]

    return words_by_freq_word


def invert_dict(d):
    return {v:k for k,v in d.items()}


def get_word2idx(full_vocab, embedded_vocab):
    """
    Creates a word->idx mapping, such that each word has an embedding
      Words that don't have embeddings (in full_vocab \setminus embedded_vocab) are embedded as UNK
      UNK is assumed to be in embedded_vocab
    :param full_vocab: list of words
    :param embedded_vocab: list of words which has embedding
    :return: word2idx
    """

    assert UNK in embedded_vocab

    # words that have embedding
    embedded_idx2word = dict(enumerate(embedded_vocab))
    word2idx = invert_dict(embedded_idx2word)

    # map words that don't have embeddings to UNK
    for wrd in full_vocab:
        if wrd not in word2idx:
            word2idx[wrd] = word2idx[UNK]

    return word2idx


def padding_sent(sent):
    return [START] + sent + [END]


def load_dataset(fname):
    """
    Return a list sentences, each sentence is a list of tokens, padding each sentence with START, END tokens
    """
    docs = []
    with open(fname, 'r', encoding='UTF-8') as fd:
        for line in fd:
            # parse sentence
            parsed = preprocess_line(line)

            # verify max len
            if len(parsed) <= MAX_SENT_LEN:
                padded = padding_sent(parsed)
                docs.append(padded)

    return docs


def load_all_data_sets(files):
    """
     Loads all sentences in the dataset
     Returns a list of lists of words
    """

    # handle the case where file is a single file
    if not isinstance(files, (list, tuple)):
        files = [files]

    files_docs = []
    for file in files:
        files_docs.append(load_dataset(file))

    return [sent for docs in files_docs for sent in docs]


def get_sents_length_vector(docs):
    lengths = []
    for sent in docs:
       lengths += [len(sent)]

    # verify MAX_SENT_LEN_W_TOKENS doesn't change
    assert MAX_SENT_LEN_W_TOKENS >= max(lengths), \
        "Sentences should not be over %d words. Something is wrong, check at load_dataset()" % MAX_SENT_LEN_W_TOKENS

    return np.array(lengths)


def pad_sent_to_max_length(sent):
    return sent + [PAD] * (MAX_SENT_LEN_W_TOKENS - len(sent))


def sent_to_integer(sent, word_to_int):
    """
    Replaces each word in sent with integer by word_to_int
    """
    return np.array([word_to_int[w] if w in word_to_int else word_to_int[UNK] for w in sent])


def sents_to_integers(sents, word_to_int):
    """
    Replaces each word with the corresponding int, unless its unknown (than gets UNK int)
    returns a numpy matrix of size MAX_SENT_LEN * dataset_size
    """
    ret = []
    for sent in sents:
        #TODO: canonoicalize???
        ret.append(sent_to_integer(pad_sent_to_max_length(sent), word_to_int))

    # return as numpy array for fancier slicing
    return np.array(ret, dtype=object)


def get_glove_word_embedding(words, glove_file, dims, eps=EPS):
    """
    Filter embeddings found in glove_file for words
    Make sure that tokens have an embedding
    :param words: list of words
    :param glove_file: path of file in glove format
    :param dims: num of embedding dimensions
    :param eps: used to generate embeddings for tokens
    :return: embedding matrix, words_left
      where the embedding of words_left[i] is at embedding[i, :]
    """

    # make sure words has consistent order
    assert isinstance(words, (list, tuple))

    # keep track of found words
    words_to_find = set(words)

    # embedding matrix to fill
    embedding = np.zeros((len(words), dims))

    # filter glove file to words
    with open(glove_file, "r", encoding="UTF-8") as glv:
        for line in glv:
            # glove file has format: "wrd v_0 v_1 v_2 .."
            wrd, vec = line.split(" ", 1)

            if wrd in words:
                embedding[words.index(wrd), :] = vec.split(" ")
                words_to_find.remove(wrd)

    # verify all tokens have embeddings
    # tokens will be embedded where its unlikely for other words to be embedded
    for tok in {UNK, START, END, PAD}.intersection(words_to_find):
        # create random point close to origin
        vec = np.random.uniform(size=dims)
        vec *= eps / np.linalg.norm(vec, ord=2)

        # add embedding
        embedding[words.index(tok), :] = vec
        words_to_find.remove(tok)

    # filter words not found in glove
    embedding = np.delete(embedding, [words.index(wrd) for wrd in words_to_find], axis=0)
    words_left = [wrd for wrd in words if wrd not in words_to_find]

    return embedding, words_left


def filter_glove_word(words, glove_file, new_glove_file):
    """
    Filter embeddings found in glove_file for words, and saves it to new_glove_file
    :param words: list of words
    :param glove_file: path of file in glove format
    """

    # filter glove file to words
    with open(glove_file, "r", encoding="UTF-8") as glv:
        with open(new_glove_file, "w", encoding="UTF-8") as new_glv:
            for line in glv:
                # glove file has format: "wrd v_0 v_1 v_2 .."
                wrd, vec = line.split(" ", 1)

                if wrd in words:
                    new_glv.writelines(line)


def get_datasets(data_dir, set_type):
    """
    Return datasets in format consistent with preprocess_dataset 'formatsets' and 'informatsets'
    :param data_dir: path to dir which contains data under dirs 'formal' and 'informal'
    :param set_type: 'formal' or 'informal'
    """
    assert set_type in ("formal", "informal")

    datasets = {k: join(data_dir, k, set_type) for k in DATASET_TYPES}
    return datasets
