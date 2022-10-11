"""
author-gh: @adithya8
editor-gh: ykl7
"""

import math 

import numpy as np
import torch
import torch.nn as nn

sigmoid = lambda x: 1/(1 + torch.exp(-x))

class WordVec(nn.Module):
    def __init__(self, V, embedding_dim, loss_func):
        super(WordVec, self).__init__()
        self.center_embeddings = nn.Embedding(num_embeddings=V, embedding_dim=embedding_dim)
        self.center_embeddings.weight.data.normal_(mean=0, std=1/math.sqrt(embedding_dim))
        self.center_embeddings.weight.data[self.center_embeddings.weight.data<-1] = -1
        self.center_embeddings.weight.data[self.center_embeddings.weight.data>1] = 1

        self.context_embeddings = nn.Embedding(num_embeddings=V, embedding_dim=embedding_dim)
        self.context_embeddings.weight.data.normal_(mean=0, std=1/math.sqrt(embedding_dim))
        self.context_embeddings.weight.data[self.context_embeddings.weight.data<-1] = -1 + 1e-10
        self.context_embeddings.weight.data[self.context_embeddings.weight.data>1] = 1 - 1e-10
        
        self.loss_func = loss_func

    def forward(self, center_word, context_word, negative_word):

        if self.loss_func == "nll":
            return self.negative_log_likelihood_loss(center_word, context_word)
        elif self.loss_func == "neg":
            return self.negative_sampling(center_word, context_word, negative_word)
        else:
            raise Exception("No implementation found for %s"%(self.loss_func))
    
    def negative_log_likelihood_loss(self, center_word, context_word):
        ### TODO(students): start
        emb_x = self.center_embeddings(center_word)
        emb_y = self.context_embeddings(context_word)

        score1 = torch.sum(torch.mul(emb_x, emb_y), dim = 1)

        score2 = torch.log(torch.sum(torch.exp(torch.matmul(emb_x, emb_y.T)), dim = 1))

        loss = torch.mean(score2 - score1)
        ### TODO(students): end
        return loss


    def negative_sampling(self, center_word, context_word, negative_word):
        ### TODO(students): start
        emb_x = self.center_embeddings(center_word)
        emb_y = self.context_embeddings(context_word)
        emb_neg_y = self.context_embeddings(negative_word)

        #positive sample pairs
        score1 = torch.log(sigmoid(torch.sum(torch.mul(emb_x, emb_y), dim = 1)))
        score2 = torch.sum(torch.log(sigmoid((-1.0)*torch.bmm(emb_neg_y, emb_x.unsqueeze(2)).squeeze())), dim = 1)

        loss = - torch.mean(score1 + score2)
        ### TODO(students): end

        return loss

    def print_closest(self, validation_words, reverse_dictionary, top_k=8):
        print('Printing closest words')
        embeddings = torch.zeros(self.center_embeddings.weight.shape).copy_(self.center_embeddings.weight)
        embeddings = embeddings.data.cpu().numpy()

        validation_ids = validation_words
        norm = np.sqrt(np.sum(np.square(embeddings),axis=1,keepdims=True))
        normalized_embeddings = embeddings/norm
        validation_embeddings = normalized_embeddings[validation_ids]
        similarity = np.matmul(validation_embeddings, normalized_embeddings.T)
        for i in range(len(validation_ids)):
            word = reverse_dictionary[validation_words[i]]
            nearest = (-similarity[i, :]).argsort()[1:top_k+1]
            print(word, [reverse_dictionary[nearest[k]] for k in range(top_k)])            