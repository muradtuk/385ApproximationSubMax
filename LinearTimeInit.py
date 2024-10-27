import numpy as np
import numba
import utils
from tqdm import tqdm
from time import time 

@numba.njit()
def count_loop(a):
    s = 0
    for i in a:
        if i != 0:
            s += 1
    return s



@numba.njit('Tuple((int64[:], int64[:], float64[:], boolean[:]))(int64[:], int64[:], int64, float64[:], int64'
            ', float64, float64, boolean[:])', fastmath=True)
def applyProcessSubroutine(S, S_prime, e, marginals, k,
                           beta, marginal_e, non_zero_indices):

    if count_loop(non_zero_indices) < k and (S_prime[e] == 0 or marginal_e >= 0):
        S[e] = 1
        S_prime[e] = 1
        marginals[e] = marginal_e
        non_zero_indices[e] = True
    else:

        # idxs = np.where(non_zero_indices)[0]
        # a_star = np.argmin(S_prime[non_zero_indices])  # utils.numbaRoll(e1, np.argmin(marginals[S_prime != 0]), -1)
        # marginal_a_star = np.min(S_prime[non_zero_indices])
        #
        # if marginal_e > (1 + beta) * marginal_a_star:
        #     S[e] = 1
        #     S_prime[e] = 1
        #     S_prime[idxs[a_star]] = 0
        #     marginals[e] = marginal_e
        #     non_zero_indices[e] = True
        #     non_zero_indices[idxs[a_star]] = False
        # Loop manually to find the index of the smallest S_prime value
        min_val = np.inf
        a_star_idx = -1

        # Iterate through non-zero indices to find min value and index
        for i in range(non_zero_indices.size):
            if non_zero_indices[i]:
                if S_prime[i] < min_val:
                    min_val = S_prime[i]
                    a_star_idx = i

        if marginal_e > (1 + beta) * min_val:
            S[e] = 1
            S_prime[e] = 1
            S_prime[a_star_idx] = 0
            marginals[e] = marginal_e
            non_zero_indices[e] = True
            non_zero_indices[a_star_idx] = False

    return S, S_prime, marginals, non_zero_indices


@numba.njit('Tuple((int64[:], int64, boolean[:]))(float64[:,:], int64, int64, float64[:], float64, float64)',
            fastmath=True)
def applyQuickSWAPNM(A, k, function_type, D, Lambda, beta):
    B = np.zeros((A.shape[0], ), dtype=np.int64)
    B_prime = np.zeros((A.shape[0], ), dtype=np.int64)
    C = np.zeros((A.shape[0], ), dtype=np.int64)
    C_prime = np.zeros((A.shape[0], ), dtype=np.int64)
    e1 = np.zeros((A.shape[0],), dtype=B.dtype)
    marginals = np.zeros((A.shape[0],), dtype=np.float64)
    non_zero_indices_B = np.zeros((A.shape[0],), dtype=np.bool_)
    non_zero_indices_C = np.zeros((A.shape[0],), dtype=np.bool_)

    for i in (range(A.shape[0])):
        base_term_B = 0
        base_term_C = 0
        if i % 1000 == 0:
            with numba.objmode(s='f8'):
                s = time()
        if function_type == 0:
            base_term_B = utils.computeLocationSummarization(A, B, D)
            marginal_B_e = utils.computeLocationSummarization(A, B + utils.numbaRoll(e1, i, -1), D)
            base_term_C = utils.computeLocationSummarization(A, C, D)
            marginal_C_e = utils.computeLocationSummarization(A, C + utils.numbaRoll(e1, i, -1), D)
        elif function_type == 1:
            base_term_B = utils.computeRevenueMaximizationCost(A, B)
            marginal_B_e = utils.computeRevenueMaximizationCost(A, B + utils.numbaRoll(e1, i, -1))
            base_term_C = utils.computeRevenueMaximizationCost(A, C)
            marginal_C_e = utils.computeRevenueMaximizationCost(A, C + utils.numbaRoll(e1, i, -1))
        elif function_type == 2:
            base_term_B = utils.computeImageSummerizationCost(A, B)
            marginal_B_e = utils.computeImageSummerizationCost(A, B + utils.numbaRoll(e1, i, -1))
            base_term_C = utils.computeImageSummerizationCost(A, C)
            marginal_C_e = utils.computeImageSummerizationCost(A, C + utils.numbaRoll(e1, i, -1))
        elif function_type == 3:
            base_term_B = utils.computeMovieRecommendationCost(A, B, Lambda, base_term_B)
            marginal_B_e = utils.computeMovieRecommendationCost(A, B + utils.numbaRoll(e1, i, -1), Lambda, base_term_B)
            base_term_C = utils.computeMovieRecommendationCost(A, C, Lambda, base_term_C)
            marginal_C_e = utils.computeMovieRecommendationCost(A, C + utils.numbaRoll(e1, i, -1), Lambda, base_term_C)
        if i % 1000 == 0:
            with numba.objmode():
                print(f'Marginal gain took {time() - s} seconds at {i}th element')
        marginal_B_e = marginal_B_e - base_term_B
        marginal_C_e = marginal_C_e - base_term_C
        if marginal_B_e > marginal_C_e:
            B, B_prime, marginals, non_zero_indices_B = applyProcessSubroutine(S=B, S_prime=B_prime, e=i,
                                                                               marginals=marginals, k=k,
                                                                               beta=beta,
                                                                               marginal_e=marginal_B_e,
                                                                               non_zero_indices=non_zero_indices_B)
        else:
            C, C_prime, marginals, non_zero_indices_C = applyProcessSubroutine(S=C, S_prime=C_prime, e=i,
                                                                               marginals=marginals, k=k,
                                                                               beta=beta,
                                                                               marginal_e=marginal_C_e,
                                                                               non_zero_indices=non_zero_indices_C)
        
    base_term_B = 0
    base_term_C = 0
    if function_type == 0:
        base_term_B = utils.computeLocationSummarization(A, B_prime, D)
        base_term_C = utils.computeLocationSummarization(A, C_prime, D)
    elif function_type == 1:
        base_term_B = utils.computeRevenueMaximizationCost(A, B_prime)
        base_term_C = utils.computeRevenueMaximizationCost(A, C_prime)
    elif function_type == 2:
        base_term_B = utils.computeImageSummerizationCost(A, B_prime)
        base_term_C = utils.computeImageSummerizationCost(A, C_prime)
    elif function_type == 3:
        base_term_B = utils.computeMovieRecommendationCost(A, B_prime, Lambda, base_term_B)
        base_term_C = utils.computeMovieRecommendationCost(A, C_prime, Lambda, base_term_C)

    if base_term_B > base_term_C:
        return B_prime, A.shape[0], non_zero_indices_B
    else:
        return C_prime, A.shape[0], non_zero_indices_C