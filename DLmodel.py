#!/usr/bin/env python3

# Author: Mohammed Almansour

'''

--setup_seed() added to just fix random seed and get consistent results to be able to do hyperparameters tuning.

--tokenise() takes out all punctuation then splits where white spaces are.

--preprocessing() tried several things but it turned out the basic preprocessing done in the main code is sufficient. tried changing all numbers to zeros, removing words that occur once or twice only and other methods but it turned out basic preprocessing was best.

--stopWords I tried several sets of stop words but the current ones worked the best.

--network() the network takes the input of size(batch, sequence length, embedding_size) and feeds it into two layers of GRU unit with  hidden_size=256. then Dropout is used with ratio=0.2. next the output is fed into two different paths, one to predict rating and the other to predict category. the one for predicting rating is just one fully connected layer with output size =2 which is the number of classes. the category prediction path consists of two fully connected layers with sizes reduced gradually to give a final category prediction output of size=5. the activation function relu was found to perform best in this network.

--convertNetOutput() takes the output from the network and gives the indices with the max value in a long tensor type. the index represents the class that is chosen by the network for that particular review.

--loss() the loss function that gave the best performance was tnn.CrossEntropyLoss()

--learning rate=0.001 was found to be the optimum value for this network

-- glove dim= 200 was found to give the best results.

-- Optimiser Adam was found to perform best

-- other methods and structures that were tried but not adopted:
different network structure: with the input connected directly to the category prediction path in addition to the GRU output. however this made the network training slower with minimal gains.


'''

"""
Algorithm design:
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import numpy as np
import random
from itertools import repeat
from collections import Counter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# import sklearn

from config import device


# to get consistent result for each run (deterministic) to be able to compare different runs results
def setup_seed(seed):
    # fix random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(47)

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################
punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""


def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """
    # remove all punctuations in sample and then split
    processed = sample.translate(str.maketrans(punctuation, " " * len(punctuation))).split()
    
    return processed


def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    '''
    words = Counter()
    n = len(sample)
    for i in range(n):
            # The sentences will be stored as a list of words/tokens
            w = sample[i]
            #sample.pop(i)
            words.update([w.lower()])  # Converting all the words to lowercase
            sample.append(w.lower())
    #print(words)
    # Removing the words that only appear once
    words = {k:v for k,v in words.items() if v>1}
    # Sorting the words according to the number of appearances, with the most common word being first
    words = sorted(words, key=words.get, reverse=True)
    # Adding padding and unknown to our vocabulary so that they will be assigned an index
    sample = ['_PAD','_UNK'] + words
    #words1 = list(words.values())
    '''
    return sample


def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

# Stop words chosen to give best results
stopWords = {"ve", "the", "is", "a", "an", "this", "d", "i", "ll", "m", "s", "t",
             "am", "be", "was", "were", "being", "are", "have", "been", "has"}

wordVectors = GloVe(name='6B', dim=200)


################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    # get the index with the highest value which is the predicted class
    # and make it a long tensor type
    ratingOutput = torch.argmax(ratingOutput, 1, keepdim=True).long()
    categoryOutput = torch.argmax(categoryOutput, 1, keepdim=True).long()

    return ratingOutput, categoryOutput


################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        hidden_size = 256  # hidden size in GRU
        word_emb_size = 200  # dim used in Glove-6B
        self.drop = tnn.Dropout(0.2) # Dropout ratio inside
        # GRU unit
        self.gru = tnn.GRU(input_size=word_emb_size, hidden_size=hidden_size, num_layers=2, batch_first=True,
                           bidirectional=True)
        # fully connected layer for
        self.vocab_emb_hid = tnn.Linear(word_emb_size, word_emb_size, bias=True)  # fc layer for word embedding
        #  output layer for predicted rating
        self.predict_rating = tnn.Linear(hidden_size * 2, 2, bias=True)  # fc layer for rating prediction
        # hidden fully connected layer for category prediction
        self.predict_cate_hid = tnn.Linear(hidden_size * 2 , hidden_size,
                                           bias=True)  # fc layer for category prediction
        # output prediction layer for category
        self.predict_cate = tnn.Linear(hidden_size, 5, bias=True)

    def forward(self, input, length):
        # calculate mean of input sequence and feed the result into a fc layer
        #input_avg = torch.mean(input, dim=1)
        #input_avg = torch.relu(self.vocab_emb_hid(input_avg))
        # use spartial dropout and feed the result into GRU network
        gru_out_all = self.gru(self.drop(input))[0]
        # take the final output only
        gru_out = gru_out_all[:, -1, :]
        # predict rating
        ratingOutput = self.predict_rating(gru_out)
        # concate word embedding and gru output and feed the result into 2 fc layers to predict category
        #category_hid_input = torch.cat([gru_out, input_avg], dim=1)
        # fully connected layer after GRU
        category_hid = torch.relu(self.predict_cate_hid(gru_out))
        # prediction layer for category
        categoryOutput = self.predict_cate(category_hid)

        return ratingOutput, categoryOutput


class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        # use CrossEntropyLoss
        self.loss_func1 = tnn.CrossEntropyLoss()
        self.loss_func2 = tnn.CrossEntropyLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        # feed both network output and target output to loss function
        # for rating
        loss_rating = self.loss_func1(ratingOutput, ratingTarget)
        # for category
        loss_cate = self.loss_func2(categoryOutput, categoryTarget)
        return loss_rating + loss_cate


net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 6
optimiser = toptim.Adam(net.parameters(), lr=0.001)
