"""
author-gh: @adithya8
editor-gh: ykl7
"""

import collections

import numpy as np
import torch
import random

np.random.seed(1234)
torch.manual_seed(1234)

# Read the data into a list of strings.
def read_data(filename):
    with open(filename) as file:
        text = file.read()
        data = [token.lower() for token in text.strip().split(" ")]
    return data

def build_dataset(words, vocab_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocab_size - 1))
    # token_to_id dictionary, id_to_taken reverse_dictionary
    vocab_token_to_id = dict()
    for word, _ in count:
        vocab_token_to_id[word] = len(vocab_token_to_id)
    data = list()
    unk_count = 0
    for word in words:
        if word in vocab_token_to_id:
            index = vocab_token_to_id[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count

    vocab_id_to_token = dict(zip(vocab_token_to_id.values(), vocab_token_to_id.keys()))
    return data, count, vocab_token_to_id, vocab_id_to_token

class Dataset:
    def __init__(self, data, unigram_prob, batch_size=128, num_skips=8, skip_window=4, neg_sample_size = 0):
        """
        @data_index: the index of a word. You can access a word using data[data_index]
        @batch_size: the number of instances in one batch
        @num_skips: the number of samples you want to draw in a window 
                (In the below example, it was 2)
        @skip_windows: decides how many words to consider left and right from a context word. 
                    (So, skip_windows*2+1 = window_size)
        """

        self.data_index=0
        self.data = data
        self.unigram_prob = unigram_prob
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window

        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window
        self.neg_sample_size = neg_sample_size
    
    def reset_index(self, idx=0):
        self.data_index=idx

    def generate_batch(self):
        """
        Write the code generate a training batch

        batch will contain word ids for context words. Dimension is [batch_size].
        labels will contain word ids for predicting(target) words. Dimension is [batch_size, 1].
        """
        
        center_word = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        context_word = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        
        # stride: for the rolling window
        stride = 1 

        ### TODO(students): start
        nword = 0
        while(self.data_index < len(self.data) and nword < self.batch_size):
            sidx = max(0, self.data_index - self.skip_window)
            eidx = min(len(self.data), self.data_index + self.skip_window + 1)

            context_candidate = []
            for i in range(sidx, eidx):
                if i != self.data_index:
                    context_candidate.append(self.data[i])

            if self.num_skips== len(context_candidate):
                context_words = context_candidate
            else:
                ncontext = min(len(context_candidate), self.num_skips)
                context_words = random.sample(context_candidate, ncontext)

            for w in context_words:
                if nword < self.batch_size:
                    center_word[nword] = self.data[self.data_index]
                    context_word[nword] = w
                    nword += 1
                else: 
                    break
            
            self.data_index += stride
        ### TODO(students): end

        """
        # Add negative sampling part for NEG loss
        """
        if self.neg_sample_size > 0:
            negative_word = np.random.choice(range(len(self.unigram_prob)), size=(len(center_word), self.neg_sample_size), p = self.unigram_prob).tolist()
        else:
            negative_word = []
        return torch.LongTensor(center_word), torch.LongTensor(context_word), torch.LongTensor(negative_word)