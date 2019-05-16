"""
Utilities for processing gensim embedding files
"""

from pathlib import Path
from gensim.models.keyedvectors import Vocab, Word2VecKeyedVectors

def read_gensim_model(file_name):
    extension = Path(file_name).suffix
    if extension == '.txt':
        model = Word2VecKeyedVectors.load_word2vec_format(file_name, binary=False)
    elif extension == '.bin' or extension == '.w2v':
        model = Word2VecKeyedVectors.load_word2vec_format(file_name, binary=True)
    else:
        raise Exception("unknown extension for embeddings file")

    return model

def save_gensim_model(words, word_reprs, output_file, binary=True):
    """Save word representations in w2v format. Word order is not preserved"""
    vocab = dict()
    for word in words:
        vocab[word] = Vocab(index=len(vocab))

    model = Word2VecKeyedVectors(word_reprs.shape[1])
    model.vocab = vocab
    model.vectors = word_reprs
    model.save_word2vec_format(fname=output_file, binary=binary)

def save_gensim_model_preserve_order(words, word_reprs, output_file):
    """Save word representations keeping the ordering provided by the words list."""
    with open(output_file, mode='w', encoding='utf8') as out:
        format_string = "%s " + ("%.6f " * (word_reprs.shape[1] - 1)) + "%.6f\n"

        out.write("%d %d\n" % (word_reprs.shape[0], word_reprs.shape[1]))
        for i in range(len(words)):
            out.write(format_string % ((words[i],) + tuple(word_reprs[i])))
