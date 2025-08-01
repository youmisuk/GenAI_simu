# this file is to load all the customized functions for this project
import numpy as np
import warnings
from collections import Counter
from nltk import ngrams


from scipy.stats import wasserstein_distance

# Function to compute unigram and bigram distributions
def compute_ngram_distribution(sequences, n=1):
    from collections import Counter
    from nltk import ngrams
    ngram_list = []
    for seq in sequences:
        tokens = seq.split(',')
        tokens = [token.strip() for token in tokens]
        ngram_list.extend(list(ngrams(tokens, n)))
    total = sum(Counter(ngram_list).values())
    return {ng: count / total for ng, count in Counter(ngram_list).items()}

# Compute JSD for N-grams
def compute_jsd(real_dist, synthetic_dist):
    all_keys = set(real_dist.keys()).union(synthetic_dist.keys())
    real_values = [real_dist.get(key, 0) for key in all_keys]
    synthetic_values = [synthetic_dist.get(key, 0) for key in all_keys]
    return jensenshannon(real_values, synthetic_values)

# Function to compute N-gram counts
def compute_ngram_counts(sequences, n):
    ngram_list = []
    for sequence in sequences:
        tokens = sequence.split(',')  # Tokenize sequence
        tokens = [token.strip() for token in tokens]  # Remove spaces
        ngram_list.extend(list(ngrams(tokens, n)))  # Generate N-grams
    return Counter(ngram_list)

# Function to return Ngram
def get_ngram_list(sequences, n):
    ngram_list = []
    for sequence in sequences:
        tokens = sequence.split(',')  # Tokenize sequence
        tokens = [token.strip() for token in tokens]  # Remove spaces
        ngram_list.extend(list(ngrams(tokens, n)))  # Generate N-grams
    return ngram_list

# Calculate ROGUE-N Precision
def calculate_rogue_precision(real_ngrams, synthetic_ngrams):
    # Count matching N-grams (intersection of real and synthetic N-grams)
    matching_ngrams = set(real_ngrams.keys()) & set(synthetic_ngrams.keys())
    matching_count = sum(synthetic_ngrams[ng] for ng in matching_ngrams)

    # Total N-grams in synthetic data
    total_synthetic_count = sum(synthetic_ngrams.values())

    # Precision calculation
    return matching_count / total_synthetic_count if total_synthetic_count > 0 else 0

# Calculate ROGUE-N Recall
def calculate_rogue_recall(real_ngrams, synthetic_ngrams):
    # Count matching N-grams (intersection of real and synthetic N-grams)
    matching_ngrams = set(real_ngrams.keys()) & set(synthetic_ngrams.keys())
    # PAN: note here for Recall you should use real_ngrams
    matching_count = sum(real_ngrams[ng] for ng in matching_ngrams)

    # Total N-grams in synthetic data
    total_real_count = sum(real_ngrams.values())

    # Precision calculation
    return matching_count / total_real_count if total_real_count > 0 else 0


# Convert distributions to aligned lists
def align_distributions(real_dist, synthetic_dist):
    all_keys = set(real_dist.keys()).union(synthetic_dist.keys())
    real_values = [real_dist.get(key, 0) for key in all_keys]
    synthetic_values = [synthetic_dist.get(key, 0) for key in all_keys]
    return real_values, synthetic_values


# write a function to get the mean squared element-wise difference between two correlation matrices
def get_mse_corr(corr1, corr2):
    warnings.warn("If the matrix is diagonally symmteric, use get_mse_corr_upper instead!")
    return np.square(corr1 - corr2).values.mean()

# since the correlation matrix diaganolly symmteric, we can only use the upper triangle part to calculate the MSE
def get_mse_corr_upper(corr1, corr2):
    warnings.warn("This function only works for diagonally symmteric matrix!")
    corr1_upper = corr1[np.triu_indices_from(corr1, k=1)]
    corr2_upper = corr2[np.triu_indices_from(corr2, k=1)]

    return np.square(corr1_upper - corr2_upper).mean()

def extract_ngrams(sequences, n):
    ngram_list = []
    for sequence in sequences:
        # Split sequence into tokens and remove spaces
        tokens = sequence.split(',')
        tokens = [token.strip() for token in tokens]
        ngram_list.extend(list(ngrams(tokens, n)))
    return Counter(ngram_list)

# Normalize counts
def normalize(counter):
    total = sum(counter.values())
    return {key: value / total for key, value in counter.items()}


def TVC_Ngram_top(real_ngrams, synthetic_ngrams):
    '''
    WARNING: USE TVC_N_v3 instead
    a function to compute the TVC-N metrics
    add a check for TOP 10 N-grams matching
    The input is the real and synthetic n-grams with top N n-grams
    '''
    warnings.warn("This function is not the correct version. Use TVC_N_v4 instead.")
    # Convert to dictionary
    dict1 = dict(real_ngrams)
    dict2 = dict(synthetic_ngrams)

    # Find common unigrams and compute the average proportion
    common_unigrams = set(dict1.keys()) & set(dict2.keys())
    common_avg_proportions = {unigram: (dict1[unigram] + dict2[unigram]) / 2 for unigram in common_unigrams}

    # Sort by the average proportion in descending order and get the top 10
    top_10_common = sorted(common_avg_proportions.items(), key=lambda x: x[1], reverse=True)[:10]

    # create a list to load the different in proportion
    prop_diff = []

    for item in top_10_common:
        # compute the difference in proportion of top 10 common ngrams
        diff_i = abs(dict1[item[0]] - dict2[item[0]])
        prop_diff.append(diff_i)

    # get the sum of the difference in proportion
    TVC_N = 1 - 0.5 * sum(prop_diff)

    return top_10_common, TVC_N


# wrap up into a function
def TVC_N_calc(real_ngrams, synthetic_ngrams):
    '''
    WARNING: USE TVC_N_v3 instead
    Function to compute the TVC_N score treating all n-grams as categorical units
    the input should be ngrams extracted from the real and synthetic data with freq not prop
    '''
    warnings.warn("This function is not the correct version. Use TVC_N_v4 instead.")

    real_ngrams = normalize(real_ngrams)
    synthetic_ngrams = normalize(synthetic_ngrams)

    dict1 = dict(real_ngrams)
    dict2 = dict(synthetic_ngrams)

    # Find common unigrams and compute the average proportion
    common_ngrams = set(dict1.keys()) & set(dict2.keys())

    # check the which ngrams are not common
    not_common_ngrams_1 = set(dict1.keys()) - common_ngrams
    not_common_ngrams_2 = set(dict2.keys()) - common_ngrams

    if len(not_common_ngrams_1) > 0:
        print("N-grams in Real Data but not in Synthetic Data.")
        
        # attach not common ngrams to dict2 with value 0
        for ngram in not_common_ngrams_1:
            dict2[ngram] = 0

    if len(not_common_ngrams_2) > 0:
        print("N-grams in Synthetic Data but not in Real Data.")
        
        # attach not common ngrams to dict1 with value 0
        for ngram in not_common_ngrams_2:
            dict1[ngram] = 0

    common_ngrams_padded = set(dict1.keys()) & set(dict2.keys())
    prop_diff = []
    for item in common_ngrams_padded:
        # compute the difference in proportion of top 10 common ngrams
        diff_i = abs(dict1[item] - dict2[item])
        prop_diff.append(diff_i)

    TVC_N = 1- 0.5 * sum(prop_diff)
    
    return TVC_N

def TVC_N_v3(real_ngrams, synthetic_ngrams, top_n = 10):
    '''
    WARNING: DROPPED!! USE V4 INSTEAD!!
    Funciton to calculate the TVC_N score: this is the incorrect version based on Professor's description.
    The input should be ngrams extracted from the real and synthetic data with freq not prop (not normalized)
    '''
    # add a warning message to indicate that this function is not the correct one
    warnings.warn("This function is the correct version. You can use either v3 or v4.")

    real_top_n = real_ngrams.most_common(top_n)
    synthetic_top_n = synthetic_ngrams.most_common(top_n)

    real_top_n = dict(real_top_n)
    synthetic_top_n = dict(synthetic_top_n)

    # change the top ten n-grams' frequency into proportion
    real_top_n_prop = {key: value/sum(real_top_n.values()) for key, value in real_top_n.items()}
    synthetic_top_n_prop = {key: value/sum(synthetic_top_n.values()) for key, value in synthetic_top_n.items()}

    dict1 = dict(real_top_n_prop)
    dict2 = dict(synthetic_top_n_prop)

    common_ngrams = set(dict1.keys()) & set(dict2.keys())

    # check the which ngrams are not common
    not_common_ngrams_1 = set(dict1.keys()) - common_ngrams
    not_common_ngrams_2 = set(dict2.keys()) - common_ngrams

    if len(not_common_ngrams_1) > 0:
        print("Some grams in Real Data but not in Synthetic Data. The later will be padded with 0.")
        
        # attach not common ngrams to dict2 with value 0
        for ngram in not_common_ngrams_1:
            dict2[ngram] = 0

    if len(not_common_ngrams_2) > 0:
        print("Some grams in Synthetic Data but not in Real Data. The later will be padded with 0.")
        
        # attach not common ngrams to dict1 with value 0
        for ngram in not_common_ngrams_2:
            dict1[ngram] = 0

    common_ngrams_padded = set(dict1.keys()) & set(dict2.keys())

    prop_diff = []
    for item in common_ngrams_padded:
        # compute the difference in proportion of top 10 common ngrams
        diff_i = abs(dict1[item] - dict2[item])
        prop_diff.append(diff_i)

    # get the sum of the difference in proportion
    TVC_N = 1 - 0.5 * sum(prop_diff)

    return TVC_N


# test the new TVC-N function

def TVC_N_v4(real_ngrams, synthetic_ngrams, top_n = 10):
    '''
    Funciton to calculate the TVC_N score: this is the correct version based on Professor's description.
    ,where I normalize the proportion of the TOP 10 N-grams and make the metric scale to [0,1]
    The input should be ngrams extracted from the real and synthetic data with freq not prop (not normalized)
    '''
    warnings.warn("This function is the correct version. You can use either v3 or v4.")
    real_top_n = real_ngrams.most_common(top_n)
    synthetic_top_n = synthetic_ngrams.most_common(top_n)

    real_top_n = dict(real_top_n)
    synthetic_top_n = dict(synthetic_top_n)

    # change the top ten n-grams' frequency into proportion
    # real_top_n_prop = {key: value/sum(real_top_n.values()) for key, value in real_top_n.items()}
    # synthetic_top_n_prop = {key: value/sum(synthetic_top_n.values()) for key, value in synthetic_top_n.items()}

    dict1 = dict(real_top_n)
    dict2 = dict(synthetic_top_n)

    common_ngrams = set(dict1.keys()) & set(dict2.keys())

    # check the which ngrams are not common
    not_common_ngrams_1 = set(dict1.keys()) - common_ngrams
    not_common_ngrams_2 = set(dict2.keys()) - common_ngrams

    if len(not_common_ngrams_1) > 0:
        print("Some grams in Real Data but not in Synthetic Data. The later will be padded with 0.")
    
    # attach not common ngrams to dict2 with value 0
    for ngram in not_common_ngrams_1:
        dict2[ngram] = 0

    if len(not_common_ngrams_2) > 0:
        print("Some grams in Synthetic Data but not in Real Data. The later will be padded with 0.")
        
        # attach not common ngrams to dict1 with value 0
        for ngram in not_common_ngrams_2:
            dict1[ngram] = 0

    common_ngrams_padded = set(dict1.keys()) & set(dict2.keys())
    
    prop_diff = []
    for item in common_ngrams_padded:
        # compute the difference in proportion of top 10 common ngrams
        diff_i = abs(dict1[item]/sum(dict1.values()) - dict2[item]/sum(dict2.values()))
        prop_diff.append(diff_i)

    # get the sum of the difference in proportion
    TVC_N = 1 - 0.5 * sum(prop_diff)

    return TVC_N