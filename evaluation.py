#!/usr/bin/env python3

import math
import argparse
import itertools

import numpy as np
import tensorflow as tf

import data


def get_target_based_rank(composed_repr, targets, max_rank, dictionary_embeddings):
    """
    Computes the ranks of the composed representations, given a dictionary of embeddings. 
    The ordering is relative to the target representation.

    :param composed_repr: a batch of composed representations
    :param targets: a batch of targets (i.e. the phrases to be composed)
    :param max_rank: the maximum rank
    :param dictionary_embeddings: a gensim model containing the original embeddings
    :return: a list with the ranks for all the composed representations in the batch 
    """
    all_ranks = []
    target_idxs = [dictionary_embeddings.wv.vocab[w].index for w in targets]
    target_repr = np.take(dictionary_embeddings.wv.syn0, target_idxs, axis=0)
    target_dict_similarities = np.dot(dictionary_embeddings.wv.syn0, np.transpose(target_repr))

    for i in range(len(composed_repr)):
        # compute similarity between the target and the predicted vector
        target_composed_similarity = np.dot(composed_repr[i], target_repr[i])

        # remove the similarity of the target vector to itself
        target_sims = np.delete(target_dict_similarities[:, i], target_idxs[i])

        # the rank is the number of vectors with greater similarity that the one between
        # the target representation and the composed one; no sorting is required, just 
        # the number of elements that are more similar
        rank = np.count_nonzero(target_sims > target_composed_similarity) + 1
        if (rank > max_rank):
            rank = max_rank
        all_ranks.append(rank)

    return all_ranks

def get_all_ranks(predictions_file, word_embeddings, max_rank, batch_size, path_to_ranks):
    """
    For a file of predictions, compute all ranks.
    :param predictions_file: a file path to the file that contains the original compounds and the predicted embeddings
    :param word_embeddings: gensim model that contains the word embeddings
    :param max_rank: the maximum rank
    :param batch_size: for how many instances the ranks should be calculated at once
    :param config: a tf.config with specified options
    :param device: which device the dot product is calculated on (cpu or gpu)
    :param path_to_ranks: a file path to were the compounds with ranks are saved
    :return:
    """
    assert max_rank <= len(word_embeddings.wv.vocab), \
        "out of bounds error: the maximum number of nearest neighbours shouldn't be larger than the vocab size"
    compound_batches, prediction_batches = read_test_data(predictions_file=predictions_file, batch_size=batch_size)
    all_ranks = []
    for batch_idx in range(len(prediction_batches)):
        ranks = get_target_based_rank(composed_repr=prediction_batches[batch_idx],
                         targets=compound_batches[batch_idx],
                         max_rank=max_rank, dictionary_embeddings=word_embeddings)
        all_ranks.append(ranks)

    # return the flattened list
    all_ranks = [y for x in all_ranks for y in x]
    if path_to_ranks != "":
        all_compounds = [y for x in compound_batches for y in x]
        save_ranks(compounds=all_compounds, ranks=all_ranks, file_to_save=path_to_ranks)
    return all_ranks

def get_loss(predictions_file, word_embeddings, batch_size):
    """ Compute the cosine distance loss between the predictions and the original word embeddings"""
    compound_batches, prediction_batches = read_test_data(predictions_file=predictions_file, batch_size=batch_size)
    loss = 0.0
    for batch_idx in range(len(prediction_batches)):
        target_idxs = [word_embeddings.wv.vocab[w].index for w in compound_batches[batch_idx]]
        targets = np.take(word_embeddings.wv.syn0, target_idxs, axis=0)
        predictions = prediction_batches[batch_idx]

        batch_loss = tf.losses.cosine_distance(labels=targets, predictions=predictions, \
                        axis=1, reduction=tf.losses.Reduction.SUM)
        loss += batch_loss
    loss /= sum(1 for it in itertools.chain(*prediction_batches))

    return loss

def read_test_data(predictions_file, batch_size):
    """
    reads in a file containing predictions and returns batches of data
    :param predictions_file: a file path to the file that contains the original compounds and the predicted embeddings
    :param batch_size: integer that specifies the size of one batch
    :return: two lists of batches, one with the predicted embeddings and one with the corresponding compounds
    """
    original_compound_batches, prediction_batches, original_compounds, predicted_embeddings = [], [], [], []
    with open(predictions_file, "r", encoding="utf8") as f:
        for line in f:
            line_parts = line.strip().split(" ")
            assert (len(line_parts) > 2), "error: wrong number of elements on line"
            # append original comound to batch
            original_compounds.append(line_parts[0])
            # append predicted embedding to batch
            predicted_embedding = np.array(line_parts[1:]).astype(np.float32)
            predicted_embeddings.append(predicted_embedding)
            if len(original_compounds) == batch_size:
                assert len(original_compounds) == len(predicted_embeddings), "error: batches need to have same size"
                original_compound_batches.append(original_compounds)
                prediction_batches.append(np.array(predicted_embeddings))
                original_compounds, predicted_embeddings = [], []
    if len(original_compounds) > 0:
        original_compound_batches.append(np.array(original_compounds))
        prediction_batches.append(np.array(predicted_embeddings))
    return original_compound_batches, prediction_batches


def save_ranks(compounds, ranks, file_to_save):
    with open(file_to_save, "w", encoding="utf8") as f:
        for i in range(len(compounds)):
            f.write(compounds[i] + " " + str(ranks[i]) + "\n")
    print("ranks saved to file: " + file_to_save)


def calculate_quartiles(data):
    """
    get the quartiles for the data
    :param data: a list of ranks
    :return: the three quartiles we are interested in
    """
    sorted_data = sorted(data)
    leq5 = sum([1 for rank in sorted_data if rank <=5])
    mid_index = math.floor((len(sorted_data) - 1) / 2)
    if len(sorted_data) % 2 != 0:
        quartiles = list(map(np.median, [sorted_data[0:mid_index], sorted_data, sorted_data[mid_index + 1:]]))
    else:
        quartiles = list(map(np.median, [sorted_data[0:mid_index + 1], sorted_data, sorted_data[mid_index + 1:]]))
    return quartiles, "%.2f%% of ranks <=5" % (100*leq5/float(len(sorted_data)))


if __name__ == '__main__':
    # define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings", type=str, help="path to the file that contains word embeddings, format:txt")
    parser.add_argument("predictions", type=str, help="path to the file that contains the predictions for the test data")
    parser.add_argument("ranks", type=str, help="path to the file were the rank for each compound is saved to", default="")
    parser.add_argument("--unknown_word_key", type=str, 
                        help="string corresponding to the unknown word embedding in the embedding file", default="<unk>")
    parser.add_argument("--max_rank", type=int, help="maximum rank", default=1000)
    parser.add_argument("--batch_size", type=int, help="how many instances per batch", default=500)
    args = parser.parse_args()

    embeddings = data.read_word_embeddings(args.embeddings, args.unknown_word_key)
    ranks = get_all_ranks(predictions_file=args.predictions, word_embeddings=embeddings,
                          max_rank=args.max_rank, batch_size=args.batch_size, path_to_ranks=args.ranks)
    print("ranks\n")
    print(sorted(ranks))
    print("quartiles\n")
    print(calculate_quartiles(ranks))

    tf.enable_eager_execution()
    loss = get_loss(predictions_file=args.predictions, word_embeddings=embeddings, \
                            batch_size=args.batch_size)
    print("loss %.5f\n" % loss)
