from collections import Counter
from pathlib import Path
from typing import List

import pandas as pd
import torchwordemb
import torch


from seutil import IOUtils
from pts.processor.data_utils.SubTokenizer import SubTokenizer

class VocabBuilder(object):
    """
    Read file and create word_to_index dictionary.
    This can truncate low-frequency words with min_sample option.
    """
    def __init__(self, path_file=None):
        # word count
        self.word_count = VocabBuilder.count_from_file(path_file)
        self.word_to_index = {}

    @staticmethod
    def count_from_file(data_file: Path) -> Counter:
        """
        count word frequencies in a file.
        Args:
            :param data_file: {input: List[str], label: int}

        Returns:
            dict: {word_n :count_n, ...}

        """
        objs = IOUtils.load_json_stream(data_file)
        word_count = Counter()
        for obj in objs:
            inputs: List[str] = obj["input"]
            for inp_seq in inputs:
                tk_list: List[str] = SubTokenizer.sub_tokenize_java_like(inp_seq)
                word_count.update(tk_list)
            # end for
        # end for
        print('Original Vocab size:{}'.format(len(word_count)))
        return word_count

    def get_word_index(self, min_sample=1, padding_marker='__PADDING__', unknown_marker='__UNK__',):
        """
        create word-to-index mapping. Padding and unknown are added to last 2 indices.

        Args:
            min_sample: for Truncation
            padding_marker: padding mark
            unknown_marker: unknown-word mark

        Returns:
            dict: {word_n: index_n, ... }

        """
        # truncate low fq word
        _word_count = filter(lambda x:  min_sample<=x[1], self.word_count.items())
        tokens = list(zip(*_word_count))[0]

        # inset padding and unknown
        self.word_to_index = {tkn: i for i, tkn in enumerate([padding_marker, unknown_marker] + sorted(tokens))}
        print('Turncated vocab size:{} (removed:{})'.format(len(self.word_to_index),
                                                            len(self.word_count) - len(self.word_to_index)+2))
        return self.word_to_index, None


class GloveVocabBuilder(object) :

    def __init__(self, path_glove):
        self.vec = None
        self.vocab = None
        self.path_glove = path_glove

    def get_word_index(self, padding_marker='__PADDING__', unknown_marker='__UNK__',):
        _vocab, _vec = torchwordemb.load_glove_text(self.path_glove)
        vocab = {padding_marker:0, unknown_marker:1}
        for tkn, indx in _vocab.items():
            vocab[tkn] = indx + 2
        vec_2 = torch.zeros((2, _vec.size(1)))
        vec_2[1].normal_()
        self.vec = torch.cat((vec_2, _vec))
        self.vocab = vocab
        return self.vocab, self.vec






if __name__ == "__main__":

    # v_builder = VocabBuilder(path_file='data/train.tsv')
    # d = v_builder.get_word_index(min_sample=10)
    # print (d['__UNK__'])
    # for k, v in sorted(d.items())[:100]:
    #     print (k,v)

    v_builder = GloveVocabBuilder()
    d, vec = v_builder.get_word_index()
    print (d['__UNK__'])
    for k, v in sorted(d.items())[:100]:
        print (k,v)
        print(v)
