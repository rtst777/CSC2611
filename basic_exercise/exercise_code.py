import os
import json
import nltk
import math
import numpy as np
from nltk.corpus import brown
from nltk.corpus import words
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import gensim
from gensim.models import KeyedVectors
from gensim import utils

TABLE2 = {
    "asylum", "gem", "autograph", "glass", "boy", "graveyard", "brother", "grin", "car", "mound", "coast", "noon",
    "cock", "oracle", "cord", "slave", "crane", "tool", "cushion", "voyage", "food", "wizard", "furnace", "woodland",
    "automobile", "midday", "bird", "monk", "cemetery", "pillow", "forest", "rooster", "fruit", "sage", "hill", "serf",
    "implement", "shore", "jewel", "signature", "journey", "smile", "lad", "stove", "madhouse", "string", "magician",
    "tumbler",
}

TABLE1 = [['cord', 'smile', 0.02],
          ['rooster', 'voyage', 0.04],
          ['noon', 'string', 0.04],
          ['fruit', 'furnace', 0.05],
          ['autograph', 'shore', 0.06],
          ['automobile', 'wizard', 0.11],
          ['mound', 'stove', 0.14],
          ['grin', 'implement', 0.18],
          ['asylum', 'fruit', 0.19],
          ['asylum', 'monk', 0.39],
          ['graveyard', 'madhouse', 0.42],
          ['glass', 'magician', 0.44],
          ['boy', 'rooster', 0.44],
          ['cushion', 'jewel', 0.45],
          ['monk', 'slave', 0.57],
          ['asylum', 'cemetery', 0.79],
          ['coast', 'forest', 0.85],
          ['grin', 'lad', 0.88],
          ['shore', 'woodland', 0.90],
          ['monk', 'oracle', 0.91],
          ['boy', 'sage', 0.96],
          ['automobile', 'cushion', 0.97],
          ['mound', 'shore', 0.97],
          ['lad', 'wizard', 0.99],
          ['forest', 'graveyard', 1.00],
          ['food', 'rooster', 1.09],
          ['cemetery', 'woodland', 1.18],
          ['shore', 'voyage', 1.22],
          ['bird', 'woodland', 1.24],
          ['coast', 'hill', 1.26],
          ['furnace', 'implement', 1.37],
          ['crane', 'rooster', 1.41],
          ['hill', 'woodland', 1.48],
          ['car', 'journey', 1.55],
          ['cemetery', 'mound', 1.69],
          ['glass', 'jewel', 1.78],
          ['magician', 'oracle', 1.82],
          ['crane', 'implement', 2.37],
          ['brother', 'lad', 2.41],
          ['sage', 'wizard', 2.46],
          ['oracle', 'sage', 2.61],
          ['bird', 'crane', 2.63],
          ['bird', 'cock', 2.63],
          ['food', 'fruit', 2.69],
          ['brother', 'monk', 2.74],
          ['asylum', 'madhouse', 3.04],
          ['furnace', 'stove', 3.11],
          ['magician', 'wizard', 3.21],
          ['hill', 'mound', 3.29],
          ['cord', 'string', 3.41],
          ['glass', 'tumbler', 3.45],
          ['grin', 'smile', 3.46],
          ['serf', 'slave', 3.46],
          ['journey', 'voyage', 3.58],
          ['autograph', 'signature', 3.59],
          ['coast', 'shore', 3.60],
          ['forest', 'woodland', 3.65],
          ['implement', 'tool', 3.66],
          ['cock', 'rooster', 3.68],
          ['boy', 'lad', 3.82],
          ['cushion', 'pillow', 3.84],
          ['cemetery', 'graveyard', 3.88],
          ['automobile', 'car', 3.92],
          ['midday', 'noon', 3.94],
          ['gem', 'jewel', 3.94]]


def save_data(directory, file_name, data):
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = os.path.join(directory, file_name)
    with open(path, 'w') as f:
        json.dump(data, f)


def load_data(directory, file_name):
    path = os.path.join(directory, file_name)
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def prepare_corpus():
    freq_words = nltk.FreqDist(brown.words())
    most_common_tokens = freq_words.most_common(10000)

    valid_english_words = set(words.words())
    most_common_english_words = [item[0] for item in most_common_tokens if item[0] in valid_english_words]
    top_5000_english_words = most_common_english_words[:5000]
    top_5 = top_5000_english_words[:5]
    bot_5 = top_5000_english_words[-5:]
    save_data("step2", "top5_common_words.json", top_5)
    save_data("step2", "bot5_common_words.json", bot_5)

    W = list(set(top_5000_english_words).union(TABLE2))
    W.sort()
    save_data("step2", "W.json", W)
    print("number of words in W: %d" % len(W))


def ppmi(word1_count, word2_counts, total_unigram_count, bigram_counts, total_bigram_count):
    prob_word1 = word1_count / total_unigram_count
    prob_word2 = word2_counts / total_unigram_count
    prob_word1_word2 = bigram_counts / total_bigram_count
    pmi = np.log2(prob_word1_word2 / (prob_word1 * prob_word2))

    ppmi = np.nan_to_num(pmi)
    ppmi[ppmi < 0] = 0.
    return ppmi


def dim_reduction(data, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)


def get_word_pairs(word_to_index):
    pair_index = np.empty((len(TABLE1), 2), dtype=int)
    human_score = np.empty(len(TABLE1))

    for i, pair in enumerate(TABLE1):
        pair_index[i][0] = word_to_index[pair[0]]
        pair_index[i][1] = word_to_index[pair[1]]
        human_score[i] = pair[2]

    return pair_index, human_score


def compute_cosine_similarity(data, pair_index):
    cosine_similarity_score = np.empty(len(pair_index))
    for i, pair in enumerate(pair_index):
        word_index1 = pair[0]
        word_index2 = pair[1]
        cosine_similarity_score[i] = cosine_similarity(data[word_index1].reshape(1, -1), data[word_index2].reshape(1, -1))[0][0]

    return cosine_similarity_score


def main():
    # Steps 2
    prepare_corpus()

    # Steps 3
    freq_words = nltk.FreqDist(brown.words())
    W = load_data("step2", "W.json")
    total_num_words = len(W)

    word_to_index = {}
    index_to_word = {}
    word_count = np.zeros(total_num_words, dtype=np.float32)
    total_unigram_count = 0.
    for index, word in enumerate(W):
        index_to_word[index] = word
        word_to_index[word] = index
        count = freq_words.get(word)
        if count is None:
            count = 0.
        total_unigram_count += count
        word_count[index] = count

    M1 = np.zeros((total_num_words, total_num_words), dtype=np.float32)
    W_set = set(W)
    bigram_words = list(nltk.bigrams(brown.words()))
    total_bigram_count = 0.
    for bigram_word in bigram_words:
        word1 = bigram_word[0]
        word2 = bigram_word[1]
        if bigram_word[0] in W_set and bigram_word[1] in W_set:
            total_bigram_count += 1
            idx1 = word_to_index[word1]
            idx2 = word_to_index[word2]
            M1[idx1][idx2] += 1

    # Steps 4
    M1_plus = np.zeros((total_num_words, total_num_words), dtype=np.float32)
    for i in range(total_num_words):
        word1_count = word_count[i]
        word2_counts = word_count
        bigram_counts = M1[i]
        PPMIs = ppmi(word1_count, word2_counts, total_unigram_count, bigram_counts, total_bigram_count)
        M1_plus[i] = PPMIs

    # Step 5
    M2_10 = dim_reduction(M1_plus, 10)
    M2_100 = dim_reduction(M1_plus, 100)
    M2_300 = dim_reduction(M1_plus, 300)

    M2_300_json_dict = {index_to_word[i]: list(M2_300[i].astype(float)) for i in range(len(M2_300))}
    save_data("step5", "lsa_300dim_vec.json", M2_300_json_dict)

    # Step 6
    pair_index, S = get_word_pairs(word_to_index)

    # Step 7
    S_M1 = compute_cosine_similarity(M1, pair_index)
    S_M1_plus = compute_cosine_similarity(M1_plus, pair_index)
    S_M2_10 = compute_cosine_similarity(M2_10, pair_index)
    S_M2_100 = compute_cosine_similarity(M2_100, pair_index)
    S_M2_300 = compute_cosine_similarity(M2_300, pair_index)

    # Step 8
    M1_pearson_correlation = pearsonr(S_M1, S)
    M1_plus_pearson_correlation = pearsonr(S_M1_plus, S)
    M2_10_pearson_correlation = pearsonr(S_M2_10, S)
    M2_100_pearson_correlation = pearsonr(S_M2_100, S)
    M2_300_pearson_correlation = pearsonr(S_M2_300, S)

    print("M1_pearson_correlation: %f" % M1_pearson_correlation[0])
    print("M1_plus_pearson_correlation: %f" % M1_plus_pearson_correlation[0])
    print("M2_10_pearson_correlation: %f" % M2_10_pearson_correlation[0])
    print("M2_100_pearson_correlation: %f" % M2_100_pearson_correlation[0])
    print("M2_300_pearson_correlation: %f" % M2_300_pearson_correlation[0])


if __name__ == "__main__":
    main()
