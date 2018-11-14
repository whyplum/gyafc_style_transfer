"""
Validate dataset preprocessing
"""

from src.data_preprocess import *
from os.path import join

DATABASE = "data"
EMBEDDING = join(DATABASE, "glove.42B.300d", "filt.glove.42B.300d.txt")
EMBEDDING_DIMS = 300 # determined by glove
DATA = join(DATABASE, "E&M")
DIRS = ["test", "train", "tune"]
FORMAL = "formal"
INFORMAL = "informal"

FORMALSETS = {k: join(DATA, k, FORMAL) for k in DATASET_TYPES}
INFORMALSETS = {k: join(DATA, k, INFORMAL) for k in DATASET_TYPES}
ALLFILES = list(FORMALSETS.values()) + list(INFORMALSETS.values())


def test_locate_data():
    """
    Verify that model data is available
    """
    try:
        for f in ALLFILES:
            _ = open(f, 'r')
    except OSError as e:
        raise OSError


def test_create_vocabulary():
    # create full vocabulary
    vocabulary = create_vocabulary(ALLFILES)

    # choose a subset which 'has embedding'
    has_embedding = vocabulary[:10]
    if UNK not in has_embedding: has_embedding.append(UNK)

    # create a word2idx map for full_vocab
    word2idx = get_word2idx(vocabulary, has_embedding)

    # all words in vocabulary has embedding
    assert len(vocabulary) == len(word2idx)

    # all embeddings map to has_embedding
    assert len(set(word2idx.values())) == len(has_embedding)


def test_filter_glove_embedding():
    # create vocabulary
    vocabulary = create_vocabulary(ALLFILES)

    # create embedding [full_glove_words == (Glove vocab) intersect (full vocab from dataset)]
    embedding, _ = get_glove_word_embedding(vocabulary, EMBEDDING, EMBEDDING_DIMS)

    assert (MAX_WORDS, EMBEDDING_DIMS) == embedding.shape, embedding.shape


def test_dataset_and_labels():

    # load and preprocess sentences
    formal_docs = load_all_data_sets(list(FORMALSETS.values()))
    informal_docs = load_all_data_sets(list(INFORMALSETS.values()))
    docs = formal_docs + informal_docs

    assert 112124 == len(docs)
    assert all([len(sent) <= MAX_SENT_LEN_W_TOKENS for sent in docs])

    # generate labels
    formal_labels = np.array([FORMAL] * len(formal_docs))
    informal_labels = np.array([INFORMAL] * len(informal_docs))
    labels = np.concatenate((formal_labels, informal_labels), axis=0)

    assert 112124 == labels.shape[0]


def test_entire_preprocess():
    res_dict = preprocess_dataset(FORMALSETS, INFORMALSETS, EMBEDDING, EMBEDDING_DIMS)

    for data_type in DATASET_TYPES:
        # sentence sizes are correct
        assert MAX_SENT_LEN_W_TOKENS == res_dict[data_type]["sentences"].shape[0]

        # number of sentences align
        assert res_dict[data_type]["sentences"].shape[1] == res_dict[data_type]["sentence_lengths"].shape[0]

        # embedding works
        max_word_idx = max(np.unique(res_dict[data_type]["sentences"]))
        max_embedding_idx = res_dict["embedding"].shape[0]
        assert max_word_idx < max_embedding_idx



