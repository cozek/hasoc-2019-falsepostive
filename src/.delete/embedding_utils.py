from tqdm import tqdm
import pandas as pd  # type: ignore
import numpy as np
import torch
import mmap
import os
import io


def make_embedding_matrix(word_to_idx, embd_vectors, words, pad_zero=True):
    """
    Create embedding matrix for a specific set of words.

    Args:
        word_to_idx (dict): mapping from word to index 
            in embd_vectors 
        embd_vectors (numpy_array): embeddings for each word
        words (list): list of words in the dataset
        pad_zero : Padding is done using a zero vector instead 
            of a random vector
    """
    embedding_size = embd_vectors.shape[1]

    final_embeddings = np.zeros((len(words), embedding_size))
    num_unks = 0
    for i, word in tqdm(enumerate(words), total=len(words)):
        if word in word_to_idx:
            final_embeddings[i, :] = embd_vectors[word_to_idx[word]]
        else:
            num_unks += 1
            embedding_i = torch.ones(1, embedding_size)
            torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i, :] = embedding_i

    if pad_zero:
        final_embeddings[0, :] = 0
    return final_embeddings


def get_num_lines(file_path):
    """Returns the number of lines in a file"""
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def load_glove_from_file(glove_filepath):
    """
    Load the GloVe embeddings 

    Args:
        glove_filepath (str): path to the glove embeddings file 
    Returns:
        word_to_index (dict), embeddings (numpy.ndarary)
    """

    word_to_index = {}
    embeddings = []
    num_lines = get_num_lines(glove_filepath)
    with open(glove_filepath, "r") as fp:
        for index, line in tqdm(enumerate(fp), total=num_lines):
            line = line.split(" ")  # each line: word num1 num2 ...
            word_to_index[line[0]] = index  # word = line[0]
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)
    return word_to_index, np.stack(embeddings)


def load_fasttext(fname):
    """loads FastTExt embedding file

    :params:
    fname: Full path of ../crawl-300d-2M-subword.vec

    :returns:
    data: dict of all the words and their embedding
    vocab: list of words in the embedding
    """

    fin = io.open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
    n, d = map(int, fin.readline().split())
    vocab = []
    data = {}
    for line in tqdm(fin, total=n):
        tokens = line.rstrip().split(" ")
        vocab.append(tokens[0])
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data, vocab
