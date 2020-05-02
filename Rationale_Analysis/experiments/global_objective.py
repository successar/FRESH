'''
Util functions providing logics for extracting global rationales from a dataset.
The input to all functions is a list of lists of importance scores (as numpy arrays), the output is arguments for extraction from each.
These can be input in a module implementing RationaleExtractor, for extract_rationale().
'''
import math
import numpy as np


def global_argsort(arr):
    '''
    :returns: tupled array of ordered indices
    '''
    n,l = arr.shape
    return [(i//l,i%l) for i in arr.reshape(1,-1).argsort()][0]


def max_unconstrained(weights, lengths, max_ratio):
    max_tokens = math.ceil(sum(lengths) * max_ratio)
    glob_sort = global_argsort(weights)
    return tuple([g[-max_tokens:] for g in glob_sort])  # indexes into sentence tokens


def max_limited_min(weights, lengths, max_ratio, min_inst_ratio):
    '''
    first fill up on min_inst_ratio from each instance, then add the rest
    :param min_inst_ratio: threshold of minimal words to extract from each instance
    '''
    n = weights.shape[0]
    glob_sort = global_argsort(weights)
    rev_glob_sort = reversed([(glob_sort[0][i], glob_sort[1][i]) for i in range(len(glob_sort[0]))])
    remaining = [math.ceil(l * min_inst_ratio) for l in lengths]
    max_tokens = math.ceil(sum(lengths) * max_ratio)
    buff = []  # buffer for at-threshold instance tokens
    tok_idcs = []  # return indices
    while max(remaining) > 0 and len(tok_idcs) < max_tokens:
        for tup in rev_glob_sort:
            n, t = tup
            if remaining[n] <= 0:
                buff.append(tup)
            else:
                tok_idcs.append(tup)
                remaining[n] -= 1
    tok_idcs.extend(buff)

    return tok_idcs[:max_tokens]


def max_limited_min_trunc(weights, lengths, max_ratio, min_inst_ratio, top_k):
    '''
    :param top_k: leave only up to this many words from each instance
    '''
    trunc_weights = np.array([trunc_arr(a, top_k) for a in weights])
    trunc_lengths = [min(l, top_k) for l in lengths]
    return max_limited_min(trunc_weights, trunc_lengths, max_ratio, min_inst_ratio)


def trunc_arr(a, k):
    kth_val = np.sort(a)[-k]
    b = a * (a > kth_val)
    return b / np.linalg.norm(b, 1)
    
    
def max_contig(weights, lengths, max_ratio, min_inst_ratio):
    '''
    dynamic programming for finding globally maximizing contiguous segments.
    we will save, for each instance, the best k-length contiguous segment for all k-s in a pre-defined range, along with its total weights, then DP our way to an optimal solution.
    '''
    budget = math.ceil(sum(lengths) * max_ratio)
    n = len(lengths)
    min_per_i = [math.ceil(l * min_inst_ratio) for l in lengths]
    glob_max = max(lengths)
    
    cumu_ws = weights.cumsum(-1)
    
    w_diffs = []  # the benefit from adding j tokens to the i-th instance
    by_k_idcs = []  # location pointer for w_diffs
    
    for i, mn, mx in zip(cumu_ws, min_per_i, lengths):
        i_ws = np.zeros(mx-mn+2)  # last is going to be pit for max calcs (eternal bad candidate)
        i_idcs = []
        for l in range(mn, mx+1):
            sums = [i[l-1]] + [i[k+l]-i[k] for k in range(mx-l)]  # checked manually, should be fine
            j = np.argmax(sums)
            i_ws[l-mn]=sums[j]
            i_idcs.append(j)
        i_ws -= i_ws[0]
        i_ws[-1] = float('-inf')
        w_diffs.append(i_ws[1:])
        by_k_idcs.append(i_idcs)
    
    rem_budg = budget - sum(min_per_i)  # remaining budget, we're starting with all mins.
    
    # greedily add one token each step based on maximal gain
    inst_pointers = np.zeros(n, dtype='int32')  # remember which length we're at for each instance
    exp_gains = np.array([d[0] for d in w_diffs])  # expected gain for adding token; will be updated each instance
    
    for _ in range(rem_budg):
        next_i = exp_gains.argmax()
        inst_pointers[next_i] += 1
        exp_gains[next_i] = w_diffs[next_i][inst_pointers[next_i]] - exp_gains[next_i]
    
    opt = []
    for i in range(n):
        # winning length for instance
        l = min_per_i[i] + inst_pointers[i]  # '+1' is implicit because inst_pointers[i] was a *candidate*
        # location for length's best ngram
        j = by_k_idcs[i][inst_pointers[i]]
        opt.append((j, j+l))
    
    return opt


