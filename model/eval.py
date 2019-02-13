import argparse
from solver import Solver, VariationalSolver
from data_loader import get_loader
from configs import get_config_from_file
from utils import Vocab, Tokenizer
import os
import pickle
from models import VariationalModels


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    kwargs = parser.parse_args()

    config = get_config_from_file(kwargs.checkpoint, mode='test')

    print('Loading Vocabulary...')
    vocab = Vocab()
    vocab.load(config.word2id_path, config.id2word_path)
    print(f'Vocabulary size: {vocab.vocab_size}')

    config.vocab_size = vocab.vocab_size

    data_loader = get_loader(
        sentences=load_pickle(config.sentences_path),
        conversation_length=load_pickle(config.conversation_length_path),
        sentence_length=load_pickle(config.sentence_length_path),
        vocab=vocab,
        batch_size=config.batch_size)

    if config.model in VariationalModels:
        solver = VariationalSolver(config, None, data_loader, vocab=vocab, is_train=False)
        solver.build()
        solver.importance_sample()
    else:
        solver = Solver(config, None, data_loader, vocab=vocab, is_train=False)
        solver.build()
        solver.test()
