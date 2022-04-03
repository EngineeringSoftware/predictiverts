import argparse
from collections import Counter, defaultdict
from dpu_utils.mlutils import Vocabulary
import heapq
import json
import logging
import numpy as np
import os
import random
import sys
import torch
from torch import nn
from pts.data.diff_utils import get_edit_keywords


START = '<sos>'
END = '<eos>'
NL_EMBEDDING_SIZE = 64
CODE_EMBEDDING_SIZE = 64
MAX_VOCAB_SIZE = 10000
CODE_EMBEDDING_PATH = ""

class CodeEmbeddingStore(nn.Module):
    def __init__(self, code_threshold, code_embedding_size, code_token_counter,
                 dropout_rate, load_pretrained_embeddings=False, static=False):
        """Keeps track of the NL and code vocabularies and embeddings."""
        super(CodeEmbeddingStore, self).__init__()
        self.__code_vocabulary = Vocabulary.create_vocabulary(tokens=code_token_counter,
                                                              max_size=MAX_VOCAB_SIZE,
                                                              count_threshold=code_threshold,
                                                              add_pad=True)
        self.__code_embedding_layer = nn.Embedding(num_embeddings=len(self.__code_vocabulary),
                                                   embedding_dim=code_embedding_size,
                                                   padding_idx=self.__code_vocabulary.get_id_or_unk(
                                                       Vocabulary.get_pad()))
        self.code_embedding_dropout_layer = nn.Dropout(p=dropout_rate)

        print('Code vocabulary size: {}'.format(len(self.__code_vocabulary)))

        self.static = static
        if self.static and not load_pretrained_embeddings:
            raise ValueError(f"A static embedding must be pretrained!")

        if load_pretrained_embeddings:
            self.initialize_embeddings()

    def initialize_embeddings(self):
        with open(CODE_EMBEDDING_PATH) as f:
            code_embeddings = json.load(f)

        code_weights_matrix = np.zeros((len(self.__code_vocabulary), CODE_EMBEDDING_SIZE))
        code_word_count = 0
        for i, word in enumerate(self.__code_vocabulary.id_to_token):
            try:
                code_weights_matrix[i] = code_embeddings[word]
                code_word_count += 1
            except KeyError:
                if self.static:
                    # in static mode, use the %UNK% embedding
                    code_weights_matrix[i] = code_embeddings["%UNK%"]
                else:
                    # in non-static mode, initialize a random embedding
                    code_weights_matrix[i] = np.random.normal(scale=0.6, size=(CODE_EMBEDDING_SIZE,))

        self.__code_embedding_layer.weight = torch.nn.Parameter(torch.FloatTensor(code_weights_matrix),
                                                                requires_grad=(not self.static))

        print('Using {} pre-trained code embeddings'.format(code_word_count))

    def get_code_embeddings(self, token_ids):
        return self.code_embedding_dropout_layer(self.__code_embedding_layer(token_ids))

    @property
    def nl_vocabulary(self):
        return self.__nl_vocabulary

    @property
    def code_vocabulary(self):
        return self.__code_vocabulary

    @property
    def nl_embedding_layer(self):
        return self.__nl_embedding_layer

    @property
    def code_embedding_layer(self):
        return self.__code_embedding_layer

    def get_padded_code_ids(self, code_sequence, pad_length):
        return self.__code_vocabulary.get_id_or_unk_multiple(code_sequence,
                                                             pad_to_size=pad_length,
                                                             padding_element=self.__code_vocabulary.get_id_or_unk(
                                                                 Vocabulary.get_pad()),
                                                             )

    def get_padded_nl_ids(self, nl_sequence, pad_length):
        return self.__nl_vocabulary.get_id_or_unk_multiple(nl_sequence,
                                                           pad_to_size=pad_length,
                                                           padding_element=self.__nl_vocabulary.get_id_or_unk(
                                                               Vocabulary.get_pad()),
                                                           )

    def get_extended_padded_nl_ids(self, nl_sequence, pad_length, inp_ids, inp_tokens):
        # Derived from: https://github.com/microsoft/dpu-utils/blob/master/python/dpu_utils/mlutils/vocabulary.py
        nl_ids = []
        for token in nl_sequence:
            nl_id = self.get_nl_id(token)
            if self.is_nl_unk(nl_id) and token in inp_tokens:
                copy_idx = inp_tokens.index(token)
                nl_id = inp_ids[copy_idx]
            nl_ids.append(nl_id)

        if len(nl_ids) > pad_length:
            return nl_ids[:pad_length]
        else:
            padding = [self.__nl_vocabulary.get_id_or_unk(Vocabulary.get_pad())] * (pad_length - len(nl_ids))
            return nl_ids + padding

    def get_code_id(self, token):
        return self.__code_vocabulary.get_id_or_unk(token)

    def is_code_unk(self, id):
        return id == self.__code_vocabulary.get_id_or_unk(Vocabulary.get_unk())

    def get_code_token(self, token_id):
        return self.__code_vocabulary.get_name_for_id(token_id)

    def get_nl_id(self, token):
        return self.__nl_vocabulary.get_id_or_unk(token)

    def is_nl_unk(self, id):
        return id == self.__nl_vocabulary.get_id_or_unk(Vocabulary.get_unk())

    def get_nl_token(self, token_id):
        return self.__nl_vocabulary.get_name_for_id(token_id)

    def get_vocab_extended_nl_token(self, token_id, inp_ids, inp_tokens):
        if token_id < len(self.__nl_vocabulary):
            return self.get_nl_token(token_id)
        elif token_id in inp_ids:
            copy_idx = inp_ids.index(token_id)
            return inp_tokens[copy_idx]
        else:
            return Vocabulary.get_unk()

    def get_nl_tokens(self, token_ids, inp_ids, inp_tokens):
        tokens = [self.get_vocab_extended_nl_token(t, inp_ids, inp_tokens) for t in token_ids]
        if END in tokens:
            return tokens[:tokens.index(END)]
        return tokens

    def get_end_id(self):
        return self.get_nl_id(END)
