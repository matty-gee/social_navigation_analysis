import sklearn
import numpy as np
import random
import pandas as pd


def fake_decisions_2d(n_trials=12):
    return np.array([np.random.choice([-1,1]) for _ in range(n_trials*2)]).reshape(-2,2).astype(int)

def random_behavior_data():      
    
    # make better to test what certain patterns would look like in the behavior
    dimension = np.array([['affil', 'affil', 'affil', 'power', 'affil', 'power', 'power',
                           'affil', 'affil', 'power', 'affil', 'power', 'power', 'power',
                           'neutral', 'neutral', 'affil', 'power', 'affil', 'power', 'power',
                           'power', 'affil', 'power', 'power', 'power', 'affil', 'affil',
                           'power', 'affil', 'power', 'power', 'power', 'affil', 'affil',
                           'neutral', 'power', 'power', 'affil', 'affil', 'affil', 'affil',
                           'power', 'affil', 'affil', 'power', 'affil', 'affil', 'affil',
                           'affil', 'affil', 'power', 'power', 'power', 'affil', 'power',
                           'affil', 'affil', 'power', 'power', 'power', 'power', 'affil']]).T
    char_role_nums = np.array([[1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 4, 2, 1, 1, 9, 9, 1, 2, 4, 2, 4, 4,
                                4, 4, 1, 2, 2, 2, 1, 5, 5, 3, 5, 3, 3, 9, 3, 3, 3, 3, 5, 5, 5, 5,
                                5, 5, 2, 2, 3, 2, 5, 5, 5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3]]).T
    char_dec_nums = np.array([[1,2, 3, 4, 1, 2, 5, 6, 7,  3,  1,  4,  8,  9,  1,  2, 10,
                                5,2, 6, 3, 4, 5, 6, 11,  7,  8,  9, 12,  1,  2,  1,  3,  2,
                                3,3, 4, 5, 6, 7, 4, 5, 6,  7,  8,  9, 10, 11,  8, 12, 10,
                                11,12, 7, 8, 9, 10, 11, 12,  9, 10, 11, 12]]).T
    # fake decisions
    button_press = np.array([np.random.choice([1,2]) for _ in range(63)])[:,np.newaxis]
    decisions    = np.array([np.random.choice([-1,1]) for _ in range(63)])[:,np.newaxis]
    random_behav = pd.DataFrame(np.hstack([dimension, char_role_nums, char_dec_nums, button_press, decisions]), 
                                columns=['dimension', 'char_role_num', 'char_decision_num', 'button_press', 'decision'])
    return random_behav
    
def random_probas(size=(60,5)):
    # generate fake probabilities
    random_values = np.random.randint(low=1, high=10, size=size)
    random_probas = random_values / np.sum(random_values, axis=1)[:,np.newaxis]
    random_probas = pd.DataFrame(random_probas, columns=[f'probas_class{b+1:02d}' for b in range(random_probas.shape[1])])
    ixs = list(range(1, random_probas.shape[0]+1))
    random.shuffle(ixs) 
    random_probas.insert(0, 'ix', ixs)
    return random_probas

def random_rdm(size=5):
    random_values = np.random.randint(low=1, high=10, size=(size,1))
    return sklearn.metrics.pairwise_distances(random_values)

def random_patterns(size=(60, 1000)):
    # right now, totally random patterns
    return np.random.randint(low=1, high=10, size=size)

def random_pattern_similarity(size=60):
    # right now, should be totally uncorrelated...
    ps = np.arctanh(np.corrcoef(random_patterns(size=(size, 1000))))
    np.fill_diagonal(ps, 1)
    return ps


