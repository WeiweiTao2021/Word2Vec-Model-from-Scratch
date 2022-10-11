"""
author-gh: @adithya8
editor-gh: ykl7
"""
## how to run?
## python word_analogy.py --model_path=./baseline_models/word2vec_nll_skip_8_batch_64_embsize_128_nsample_0.model --loss_model=nll --output_filepath=./data/word_analogy_nll_skip_8.txt
## python word_analogy.py --model_path=./baseline_models --loss_model=neg --output_filepath=./data/word_analogy_neg.txt
## ./evaluate_word_analogy.pl ./data/word_analogy_dev_mturk_answers.txt ./data/word_analogy_nll_skip_8.txt ./data/word_analogy_nll_skip_8_score.txt
## ./evaluate_word_analogy.pl ./data/word_analogy_dev_mturk_answers.txt ./data/word_analogy_neg.txt ./data/word_analogy_neg_score.txt

import os
import pickle
import numpy as np
import argparse
import torch

np.random.seed(1234)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./baseline_models', help='Base directory of folder where models are saved')
    parser.add_argument('--input_filepath', type=str, default='./data/word_analogy_dev.txt', help='Word analogy file to evaluate on')
    parser.add_argument('--output_filepath', type=str, required=True, help='Predictions filepath')
    parser.add_argument("--loss_model", help="The loss function for training the word vector", default="nll", choices=["nll", "neg"])
    args, _ = parser.parse_known_args()
    return args

def read_data(file_path):
    with open(file_path,'r') as f:
        data = f.readlines()
    
    candidate, test = [], []
    for line in data:
        a, b = line.strip().split("||")
        a = [i[1:-1].split(":") for i in a.split(",")]
        b = [i[1:-1].split(":") for i in b.split(",")]
        candidate.append(a)
        test.append(b)
    
    return candidate, test

def get_embeddings(examples, embeddings):

    """
    For the word pairs in the 'examples' array, fetch embeddings and return.
    You can access your trained model via dictionary and embeddings.
    dictionary[word] will give you word_id
    and embeddings[word_id] will return the embedding for that word.

    word_id = dictionary[word]
    v1 = embeddings[word_id]

    or simply

    v1 = embeddings[dictionary[word_id]]
    """

    norm = np.sqrt(np.sum(np.square(embeddings),axis=1,keepdims=True))
    normalized_embeddings = embeddings/norm

    embs = []
    for line in examples:
        temp = []
        for pairs in line:
            temp.append([ normalized_embeddings[dictionary[pairs[0]]], normalized_embeddings[dictionary[pairs[1]]] ])
        embs.append(temp)

    result = np.array(embs)
    
    return result

def evaluate_pairs(candidate_embs, test_embs):

    """
    Write code to evaluate a relation between pairs of words.
    Find the best and worst pairs and return that.
    """

    best_pairs = []
    worst_pairs = []

    print(candidate_embs.shape)
    print("in evaluate pairs")
    ### TODO(students): start
    # average different in embeddings
    for i in range(len(candidate_embs)):
        total_diff = np.zeros(len(candidate_embs[1][1][1]))
        for candi in candidate_embs[i]:
            total_diff += candi[1] - candi[0]
        avg_diff = total_diff/len(candidate_embs)

        similarity = []
        for test in test_embs[i]:
            diff = test[1] - test[0]
            cos_similarity = np.dot(diff, avg_diff)/np.linalg.norm(avg_diff) * np.linalg.norm(diff)
            similarity.append(cos_similarity)

        best_idx = np.argmax(similarity)
        worst_idx = np.argmin(similarity)

        best_pairs.append(best_idx)
        worst_pairs.append(worst_idx)

    ### TODO(students): end
    #print(best_pairs)
    #print(worst_pairs)
    return best_pairs, worst_pairs

def write_solution(best_pairs, worst_pairs, test, path):

    """
    Write best and worst pairs to a file, that can be evaluated by evaluate_word_analogy.pl
    """
    
    ans = []
    for i, line in enumerate(test):
        temp = [f'"{pairs[0]}:{pairs[1]}"' for pairs in line]
        temp.append(f'"{line[worst_pairs[i]][0]}:{line[worst_pairs[i]][1]}"')
        temp.append(f'"{line[best_pairs[i]][0]}:{line[best_pairs[i]][1]}"')
        ans.append(" ".join(temp))

    with open(path, 'w') as f:
        f.write("\n".join(ans))


if __name__ == '__main__':

    args = parse_args()

    loss_model = args.loss_model
    model_path = args.model_path
    input_filepath = args.input_filepath

    print(f'Model file: {model_path}')
    #model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))
    model_filepath = model_path

    dictionary, embeddings = pickle.load(open(model_filepath, 'rb'))

    candidate, test = read_data(input_filepath)

    candidate_embs = get_embeddings(candidate, embeddings)

    test_embs = get_embeddings(test, embeddings)

    best_pairs, worst_pairs = evaluate_pairs(candidate_embs, test_embs)

    out_filepath = args.output_filepath
    print(f'Output file: {out_filepath}')
    write_solution(best_pairs, worst_pairs, test, out_filepath)