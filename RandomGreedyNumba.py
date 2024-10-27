import numpy as np
import numba
import utils


@numba.njit('Tuple((int64[:], int64))(float64[:,:], int64[:], int64, int64, float64[:], float64, float64, float64,'
            'boolean, boolean, int64)',
            fastmath=True)
def applyRandomGreedy(A, Z, k, function_type, D, Lambda, t_s, epsilon, faster, generate_full_marginals, us_lazy_init):
    S = np.zeros((A.shape[0], ), dtype=np.int64)
    oracle_counter = 0

    for i in range(k):
        base_term = 0
        if function_type == 0:
            base_term = utils.computeLocationSummarization(A, S, D)
        elif function_type == 1:
            base_term = utils.computeRevenueMaximizationCost(A, S)
        elif function_type == 2:
            base_term = utils.computeImageSummerizationCost(A, S)
        elif function_type == 3:
            base_term = utils.computeMovieRecommendationCost(A, S, Lambda, base_term)
        else:
            raise ValueError("Please implement your desired function")
        oracle_counter += 1

        if i < np.ceil(t_s * k):
            indices = np.where((S + Z) == 0)[0]
        else:
            indices = np.where(S == 0)[0]

        marginals = utils.computeMarginalGain(A=A, indices=indices, S=S, Lambda=Lambda, D=D,
                                              function_type=function_type)
        oracle_counter += indices.shape[0]

        k_indices = np.argsort(-(marginals[indices] - base_term))[:k]
        chosen = utils.random_sample_set(k_indices, k=1)[0]
        S[indices[chosen]] = 1

    return S, oracle_counter