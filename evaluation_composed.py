#!/usr/bin/env python3

import argparse

import numpy as np
import tensorflow as tf

import evaluation
import data

"""
WARNING! DO NOT USE! 
Provided only for comparison to previous research. 
The correct evaluation is the one in evaluation.py.
"""

def get_composed_based_rank(composed_repr, targets, max_rank, dictionary_embeddings):
    """
    Computes the ranks of the composed representations, given a dictionary of embeddings.
    The ordering is relative to the composed representation.

    :param composed_repr: a batch of composed representations
    :param targets: a batch of targets (i.e. the phrases to be composed)
    :param max_rank: the maximum rank
    :param dictionary_embeddings: a gensim model containing the original embeddings
    :return: a list with the ranks for all the composed representations in the batch
    """
    all_ranks = []
    target_idxs = [dictionary_embeddings.wv.vocab[w].index for w in targets]
    composed_dict_similarities = np.dot(dictionary_embeddings.wv.syn0, np.transpose(composed_repr))
    for i in range(len(composed_repr)):
        # count the number of vectors with greater similarity that the one between the composed representation
        # and the target one (the number of elements that are more similar)
        composed_sims = composed_dict_similarities[:, i]
        sim_composed_target = composed_sims[target_idxs[i]]
        rank = np.count_nonzero(composed_sims > sim_composed_target) + 1

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
    compound_batches, prediction_batches = evaluation.read_test_data(predictions_file=predictions_file, batch_size=batch_size)
    all_ranks = []
    for batch_idx in range(len(prediction_batches)):
        ranks = get_composed_based_rank(composed_repr=prediction_batches[batch_idx],
                         targets=compound_batches[batch_idx],
                         max_rank=max_rank, dictionary_embeddings=word_embeddings)
        all_ranks.append(ranks)

    # return the flattened list
    all_ranks = [y for x in all_ranks for y in x]
    if path_to_ranks != "":
        all_compounds = [y for x in compound_batches for y in x]
        evaluation.save_ranks(compounds=all_compounds, ranks=all_ranks, file_to_save=path_to_ranks)
    return all_ranks


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
    print(evaluation.calculate_quartiles(ranks))

    tf.enable_eager_execution()
    loss = evaluation.get_loss(predictions_file=args.predictions, word_embeddings=embeddings, \
                            batch_size=args.batch_size)
    print("loss %.5f\n" % loss)
