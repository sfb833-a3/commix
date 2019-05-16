#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import time
import logging
import csv

import tensorflow as tf
from keras.utils import generic_utils
import numpy as np

import data
from utils import matrix_mapping
from utils.ops import init_hashtable
import logger_config
import evaluation
from training_graph import TrainingGraph, RunMode
from models import Addition, \
    BiLinear, \
    FullLex, \
    Matrix, \
    ScalarWeightedAddition, \
    TransWeight, \
    TransWeightFeatures, \
    TransWeightMatrix, \
    TransWeightTransformations, \
    VectorWeightedAddition, \
    WMask

'''
Script for training a composition model. Every composition model that inherits from AbstractModel
can be trained using it.
'''

def train(args, composition_model, training_data, validation_data, sess):
    train_losses = []
    validation_losses = []
    reg_losses = []

    td = training_data
    vd = validation_data

    assert (len(td.modifier_batches) == len(td.head_batches) == len(td.compound_batches)), "error: inconsistent training batches"
    assert (len(vd.modifier_batches) == len(vd.head_batches) == len(vd.compound_batches)), "error: inconsistent validation batches"
    assert (td.no_batches != 0), "error: no training data"
    assert (vd.no_batches != 0), "error: no validation data"

    lowest_loss = float("inf")
    best_epoch = 0
    epoch = 1
    current_patience = 0
    tolerance = 1e-5

    # write information for tensorboard
    summary = tf.summary.merge_all()
    with sess:

        with tf.variable_scope("model", reuse=None):
            composition_model.create_architecture()
            train_model = TrainingGraph(composition_model=composition_model,
                                       batch_size=None,
                                       learning_rate=args.learning_rate,
                                       run_mode=RunMode.training,
                                       alpha=args.regularization)
        with tf.variable_scope("model", reuse=True):
            validation_model = TrainingGraph(composition_model=composition_model,
                                            batch_size=None,
                                            learning_rate=args.learning_rate,
                                            run_mode=RunMode.validation)

        if args.tensorboard != '':
            writer = tf.summary.FileWriter(args.tensorboard_path, sess.graph)

        # init all variables
        sess.run(tf.global_variables_initializer(), 
            feed_dict={train_model.model.lookup_init:lookup_table,
            validation_model.model.lookup_init: lookup_table})

        saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=0)
        while current_patience < args.patience:
            train_loss = 0.0
            validation_loss = 0.0
            reg_loss = 0.0

            for tidx in range(td.no_batches):
                assert (td.modifier_batches[tidx].shape 
                    == td.head_batches[tidx].shape 
                    == td.compound_batches[tidx].shape), "error: each batch has to have the same shape"
                assert (td.modifier_batches[tidx].shape != ()), "error: funny shaped batch"

                pb = generic_utils.Progbar(td.no_batches)
                # only executed if the user wants to use tensorboard

                # calculate loss for each batch for each epoch
                tloss, rloss, _ = sess.run(
                    [train_model.loss, train_model.reg_loss, train_model.train_op],
                    feed_dict={train_model.is_training: True,
                               train_model.original_vector: td.compound_batches[tidx],
                               train_model.model._u: td.modifier_batches[tidx],
                               train_model.model._v: td.head_batches[tidx]})

                train_loss += tloss
                reg_loss +=rloss
                pb.update(tidx + 1)

            for vidx in range(vd.no_batches):
                pb = generic_utils.Progbar(vd.no_batches)
                vloss, = sess.run(
                    [validation_model.loss],
                    feed_dict={validation_model.original_vector: vd.compound_batches[vidx],
                               validation_model.model._u: vd.modifier_batches[vidx],
                               validation_model.model._v: vd.head_batches[vidx]})
                
                validation_loss += vloss
                pb.update(vidx + 1)

            train_loss /= td.total_size
            validation_loss /= vd.total_size

            if (lowest_loss - validation_loss > tolerance):
                lowest_loss = validation_loss
                best_epoch = epoch
                saver.save(sess, args.model_path)
                current_patience = 0
            else:
                current_patience += 1

            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
            reg = True if args.regularization > 0.0 else False
            if reg:
                reg_loss /= td.total_size
                reg_losses.append(reg_loss)
                logger.info("(%d) epoch %d - train loss: %.5f reg loss: %.5f validation loss: %.5f" % (
                current_patience, epoch, train_loss, reg_loss, validation_loss))
            else:
                logger.info("(%d) epoch %d - train loss: %.5f validation loss: %.5f" % (current_patience, epoch, train_loss, validation_loss))
            epoch += 1
        if args.plot:
            get_plots(train_losses, reg_losses, validation_losses, epoch, reg)
        write_loss(train_losses, reg_losses, validation_losses, reg)
    return (train_losses, validation_losses, lowest_loss, best_epoch, saver)

def predict(args, data, composition_model, lookup_table, sess):
    predictions = []

    with sess:
        with tf.variable_scope("model", reuse=True):
            best_model = TrainingGraph(composition_model=composition_model,
                                            batch_size=None,
                                            learning_rate=args.learning_rate,
                                            run_mode=RunMode.validation)

            sess.run(tf.variables_initializer([best_model.model.lookup]), 
                feed_dict={best_model.model.lookup_init:lookup_table})

            logger.info("Generating predictions...")
            loss = 0
            for idx in range(data.no_batches):
                pb = generic_utils.Progbar(data.no_batches)
                batch_predictions, batch_loss = sess.run(
                    [best_model.predictions, best_model.loss],
                    feed_dict={best_model.original_vector: data.compound_batches[idx],
                               best_model.model._u: data.modifier_batches[idx],
                               best_model.model._v: data.head_batches[idx]})
                predictions.extend(batch_predictions)
                loss += batch_loss
                pb.update(idx + 1)
            loss /= data.total_size

    logger.info("Predictions generated.")
    return np.vstack(predictions), loss

def save_predictions(predictions, data, output_file):
    out = open(output_file, mode="w", encoding="utf8")
    format_str = "%s " + "%.7f " * (predictions.shape[1] - 1) + "%.7f\n"
    
    for i in range(predictions.shape[0]):
        tup = (data.text_compounds[i], ) + tuple(predictions[i])
        out.write(format_str % tup)

    logger.info("Predictions saved to %s" % output_file)
    out.close()


def get_plots(train_losses, reg_losses, validation_losses, epoch, reg):
    from utils import plot_utils
    plot_path = str(Path(args.save_path).joinpath(args.save_name + "_plot.pdf"))
    plot_utils.create_plots(train_losses, reg_losses, validation_losses, epoch, reg, plot_path)
    logger.info("Plots generated.")
    logger.info("Plots saved to %s" % plot_path)


def write_loss(training_loss, regularized_loss, validation_loss, reg):
    output_file = str(Path(args.save_path).joinpath(args.save_name + "_losses.csv"))
    with open(output_file, mode="w", encoding="utf8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if reg:
            writer.writerow(["epoch", "training_loss", "regularized_loss", "validation_loss"])
            for i in range(len(training_loss)):
                writer.writerow([i+1, training_loss[i], regularized_loss[i], validation_loss[i]])
        else:
            writer.writerow(["epoch", "training_loss", "validation_loss"])
            for i in range(len(training_loss)):
                writer.writerow([i + 1, training_loss[i], validation_loss[i]])
    logger.info("losses saved to %s" % output_file)

def do_eval(logger, args, split, loss, word_embeddings):
    logger.info("%s loss %.5f" % (split, loss))
    predictions_file = str(Path(args.save_path).joinpath(args.save_name + "_%s_predictions.txt" % split))
    ranks_file = str(Path(args.save_path).joinpath(args.save_name + "_%s_ranks.txt" % split))

    ranks = evaluation.get_all_ranks(predictions_file=predictions_file, word_embeddings=word_embeddings,
        max_rank=args.max_rank, batch_size=args.eval_batch_size, path_to_ranks=ranks_file)
    logger.info("%s quartiles" % split)
    logger.info(evaluation.calculate_quartiles(ranks))

if __name__ == '__main__':
    #define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings", type=str, help="path to the file that contains word embeddings, format: .bin/.txt")
    parser.add_argument("data_dir", type=str, help="path to the directory that contains the train/test/dev data")
    parser.add_argument("--unknown_word_key", type=str, 
                        help="string corresponding to the unknown word embedding in the embedding file", default="<unk>")
    parser.add_argument("--separator", type=str, help="separator that separates modifier, head and phrase in the datasets", default=" ")
    parser.add_argument("--composition_model", type=str,
                        choices=["addition", "bilinear", "scalar_addition", "vector_addition",
                                 "matrix", "fulllex", "wmask", "trans_weight", "trans_weight_transformations",
                                 "trans_weight_features", "trans_weight_matrix"],
                        help="which type of composition model should be used", default="vector_addition")
    parser.add_argument("--batch_size", type=int, help="how many instances should be contained in one batch?", default=100)
    parser.add_argument("--dropout", type=float, help="dropout rate", default=0.5)
    parser.add_argument("--dropout2", type=float, help="second dropout rate", default=0.5)
    parser.add_argument("--patience", type=int, help="number of epochs to wait after the best model", default=5)
    parser.add_argument("--learning_rate", type=float, help="learning rate for optimization", default=0.01)
    parser.add_argument("--tensorboard", type=str, help="if defined the information for tensorboard will be saved to the given directory", default='')
    parser.add_argument("--seed" , type=int, help="number to which random seed is set", default=1)
    parser.add_argument("--save_path", type=str, help="file path to save the best model", default="./trained_models")
    parser.add_argument("--transforms", type=int, help="number of transforms", default=120)
    parser.add_argument("--use_weighting", help="set flag to true to use weighting in TransformationWeighting model", action='store_true', default=False )
    parser.add_argument("--nonlinearity", type=str, help="what kind of nonlinear function should be applied to the model. set to 'identity' if no nonlinearity should be applied",
                        default='identity', choices=["tanh", "identity", "relu"])
    parser.add_argument("--selection_func", type=str,
                        help="the kind of selection function that should be applied any of the cluster selection models",
                        choices=["constant", "softmax", "logistic"], default="softmax")
    parser.add_argument("--regularization", type=float, help="Set to some value > 0 in order to use regularization",
                        default=0.0)
    parser.add_argument("--regularizer", type=str, help="either choose l1_regularizer or dot_regularizer", default="l1_regularizer", choices=["l1_regularizer", "dot_regularizer"])
    parser.add_argument("--plot", help="set to true to retrieve a plot of the losses", action='store_true', default=False)
    parser.add_argument("--eval_on_test", help="set flag to true to evaluate on the test set", action='store_true', default=False)
    parser.add_argument("--max_rank", type=int, help="maximum rank in rank evaluation", default=1000)
    parser.add_argument("--eval_batch_size", type=int, help="how many instances per eval batch", default=500)
    parser.add_argument("--use_nn", help="if enabled, trained matrices / vectors from nearest neighbours are used during prediction, if the actual word hasn't been trained",
                        action='store_true', default = False)
    args = parser.parse_args()

    # log cpu/gpu info, prevent allocating so much memory
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    args.config = config

    args.training_data = str(Path(args.data_dir).joinpath("train_text.txt"))
    args.validation_data = str(Path(args.data_dir).joinpath("dev_text.txt"))
    args.test_data = str(Path(args.data_dir).joinpath("test_text.txt"))

    # generate save name for the model, using timestamp and model arguments
    save_name_xtra_args = {
        "addition": "",
        "scalar_addition": "",
        "vector_addition": "",
        "matrix": "dr%.2f" % (args.dropout),
        "wmask": "dr%.2f" % (args.dropout),
        "bilinear": "dr1%.2f_dr2%.2f" % (args.dropout, args.dropout2),
        "fulllex": "dr%.2f" % (args.dropout),
        "trans_weight": "dr%.2f_tr%d" % (args.dropout, args.transforms),
        "trans_weight_transformations": "dr%.2f_tr%d" % (args.dropout, args.transforms),
        "trans_weight_features": "dr%.2f_tr%d" % (args.dropout, args.transforms),
        "trans_weight_matrix": "dr%.2f_tr%d" % (args.dropout, args.transforms),
    }

    ts = time.gmtime()
    args.save_name = format("%s_%s_%s" % (args.composition_model, time.strftime("%Y-%m-%d-%H_%M_%S", ts), \
        save_name_xtra_args[args.composition_model]))

    # check if user wants to use default save_path and if the directory already exists, create default dir if it doesn't exist
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    # setup logging
    args.log_file = str(Path(args.save_path).joinpath(args.save_name + "_log.txt"))
    logging.config.dictConfig(logger_config.create_config(args.log_file))
    logger = logging.getLogger("train")

    logger.info("Training %s composition model. Logging to %s" % (args.composition_model, args.log_file))
    logger.info("Arguments")
    for k,v in vars(args).items():
        logger.info("%s: %s" % (k, v))
    
    #read in the wordembeddings
    gensim_model = data.read_word_embeddings(args.embeddings, args.unknown_word_key)
    logger.info("Read embeddings from %s." % args.embeddings)

    #generate batches from data
    word2index = gensim_model.wv.vocab
    training_data = data.generate_instances(args.batch_size, args.training_data, word2index, args.separator, args.unknown_word_key)
    logger.info("%d training batches" % training_data.no_batches)
    logger.info("the train dictionary contains %d words" % len(training_data.mh_set))

    mh_index_map, unk_matrix_id = matrix_mapping.create_matrix_mapping(train_mh=training_data.mh_set,
                                                                       unk_vec_id=training_data.unk_vector_id)

    validation_data = data.generate_instances(args.batch_size, args.validation_data, word2index, args.separator, args.unknown_word_key)
    logger.info("%d validation batches" % validation_data.no_batches)
    test_data = data.generate_instances(args.batch_size, args.test_data, word2index, args.separator, args.unknown_word_key)
    logger.info("%d test batches" % test_data.no_batches)
    lookup_table = gensim_model.wv.syn0
    embedding_size = int(lookup_table.shape[1])
    logger.info("Batches have been generated using the data from %s" % args.data_dir)


    # maps the composition model names to corresponding classes
    composition_models = {
        "addition": Addition, 
        "bilinear": BiLinear,
        "scalar_addition": ScalarWeightedAddition,
        "vector_addition": VectorWeightedAddition,
        "matrix": Matrix,
        "fulllex": FullLex,
        "wmask" : WMask,
        "trans_weight": TransWeight,
        "trans_weight_transformations": TransWeightTransformations,
        "trans_weight_features": TransWeightFeatures,
        "trans_weight_matrix": TransWeightMatrix
    }
    nonlinear_functions = {
        "tanh": tf.nn.tanh,
        "identity": tf.identity,
        "relu": tf.nn.relu
    }
    selection_functions = {
        "constant": tf.identity,
        "softmax": tf.nn.softmax,
        "logistic": tf.nn.sigmoid
    }

    sess = tf.Session(config=args.config)

    if args.composition_model == "bilinear":
        composition_model = BiLinear(embedding_size=embedding_size, nonlinearity=nonlinear_functions[args.nonlinearity],
                                          dropout_bilinear_forms=args.dropout,
                                          dropout_matrix=args.dropout2)
    elif args.composition_model == "fulllex":
        composition_model = FullLex(embedding_size=embedding_size, mh_index_map=mh_index_map,
                                                    unk_matrix_id = unk_matrix_id,
                                                    nonlinearity=nonlinear_functions[args.nonlinearity],
                                                    dropout_rate=args.dropout,
                                                    regularizer=args.regularizer)
        init_hashtable(mh_index_map, composition_model.index_hash, sess)
    elif "trans_weight" in args.composition_model :

        composition_model = composition_models[args.composition_model](embedding_size=embedding_size,
                                                                    nonlinearity=nonlinear_functions[args.nonlinearity],
                                                                    dropout_rate=args.dropout,
                                                                    transforms=args.transforms)
    elif args.composition_model == "wmask":
        composition_model = composition_models[args.composition_model](
                                                    embedding_size=embedding_size,
                                                    mh_index_map=mh_index_map,
                                                    unk_matrix_id = unk_matrix_id,
                                                    nonlinearity=nonlinear_functions[args.nonlinearity],
                                                    dropout_rate=args.dropout)
        init_hashtable(mh_index_map, composition_model.index_hash, sess)
    elif args.composition_model == "matrix":
        composition_model = Matrix(embedding_size=embedding_size, nonlinearity=nonlinear_functions[args.nonlinearity],
                                                   dropout_rate=args.dropout)
    else:
        composition_model = composition_models[args.composition_model](embedding_size=embedding_size)

    args.model_path = str(Path(args.save_path).joinpath(args.save_name))


    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    tf.set_random_seed(args.seed)
    saver = None
    if composition_model.is_trainable:
        train_losses, validation_losses, best_loss, best_epoch, saver = train(args, composition_model, training_data, validation_data, sess)
        logger.info("Training ended. Best epoch: %d, best loss: %.3f" % (best_epoch, best_loss))
    else:
        composition_model.create_architecture()

    valid_sess = tf.Session(config=args.config)
    logger.info("Loading best model from %s" % args.model_path)
    if saver != None:
        saver.restore(valid_sess, args.model_path)
    if args.use_nn:
        dev_index_map = matrix_mapping.create_matrix_mapping_with_neighbours(validation_data.mh_set, gensim_model, mh_index_map)
        init_hashtable(dev_index_map, composition_model.index_hash, valid_sess)

    dev_predictions, dev_loss = predict(args, validation_data, composition_model, lookup_table, valid_sess)
    save_predictions(dev_predictions, validation_data, 
        str(Path(args.save_path).joinpath(args.save_name + "_dev_predictions.txt")))
    # eval on dev
    do_eval(logger=logger, args=args, split="dev", loss=dev_loss, word_embeddings=gensim_model)
    test_sess = tf.Session(config=args.config)
    logger.info("Loading best model from %s" % args.model_path)
    if saver != None:
        saver.restore(test_sess, args.model_path)
    if args.use_nn:
        test_index_map = matrix_mapping.create_matrix_mapping_with_neighbours(test_data.mh_set, gensim_model, mh_index_map)
        init_hashtable(test_index_map, composition_model.index_hash, test_sess)

    test_predictions, test_loss = predict(args, test_data, composition_model, lookup_table, test_sess)
    save_predictions(test_predictions, test_data, 
        str(Path(args.save_path).joinpath(args.save_name + "_test_predictions.txt")))    
    if args.eval_on_test:
        do_eval(logger=logger, args=args, split="test", loss=test_loss, word_embeddings=gensim_model)
