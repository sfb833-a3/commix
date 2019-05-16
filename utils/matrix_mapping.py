import numpy as np


def create_matrix_mapping(train_mh, unk_vec_id):
    """
    Creates a map from the vector index to the matrix index, thus one can lookup the trained matrices for the words
    that occur in the training data
    :param train_mh: the set that contains all word indices of the training data
    :return: mapping of word index : matrix index
    """
    mh_index_map = {}
    matrix_idx = 0
    for vector_idx in train_mh:
        if vector_idx == unk_vec_id:
            unk_matrix_id = matrix_idx
        mh_index_map[vector_idx] = matrix_idx
        matrix_idx += 1
    return mh_index_map, unk_matrix_id

def create_matrix_mapping_with_neighbours(data_mh, embedding_model, train_mh_index_map):
    """
    Creates a map from the vector index to the matrix index, thus one can lookup the trained matrices for the words
    that occur in the training data. If a word does not occur in training data, the matrix of the nearest neighbour is
    being mapped to the word index. The nearest neighbours are calculated for all modifier or heads that do not occur
    in trainingdata.
    :param data_mh: the indices of all modifier and heads of the current dataset
    :param embedding_model: the vocabulary
    :param train_mh_index_map: the indices of all modifier and heads that occur in the training data
    :return: mapping of word index: matrix index
    """
    mh_index_map = {}
    # extract a matrix that contains only embeddings of training data
    train_vectors = embedding_model.wv.syn0[np.array(list(train_mh_index_map.keys()))]
    # dictionary: index in small matrix - real word index
    vector_index_map = dict(zip(np.arange(0, train_vectors.shape[0]), list(train_mh_index_map.keys())))
    for vector_idx in data_mh:
        # look if an instance is in train
        if vector_idx not in train_mh_index_map.keys():
            # get the vector
            word_vec = embedding_model.wv.syn0[vector_idx]
            # calculate the dot product between the current vector and vectors from train
            similarities = np.dot(train_vectors, word_vec)
            # get the one with highest similarity
            nearest = np.argmax(similarities)
            # lookup the wordindex based on the vector index of the nearest neighbour of that small matrix
            nearest_idx = vector_index_map[nearest]
            # lookup the matrix idx of that word and put it in the map
            nearest_n_matrix_idx = train_mh_index_map[nearest_idx]
            mh_index_map[vector_idx] = nearest_n_matrix_idx
        else:
            matrix_idx = train_mh_index_map[vector_idx]
            mh_index_map[vector_idx] = matrix_idx
    return mh_index_map
