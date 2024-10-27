import numpy as np
import utils
import numba


@numba.njit('Tuple((int64[:], int64, float64[:]))(float64[:,:], int64[:], int64, int64, float64[:], float64,'
            ' float64, float64, boolean, boolean, int64)', fastmath=True)
def applyLazierThanLazyGreedy(A, Z, k, function_type, D, Lambda, t_s, epsilon, faster=False,
                              generate_full_marginals=False, us_lazy_init=1):
    S = np.zeros((A.shape[0],), dtype=np.int64)
    p = 8 / k / epsilon # 8 / k / epsilon ** 2 * np.log(2 / epsilon)
    oracle_counter = 0
    global_marginals = np.zeros((A.shape[0],), dtype=np.float64)
    full_indices = np.arange(A.shape[0], dtype=np.int64)
    p = min(p, 1)

    if Z is not None and not (np.isscalar(Z) and Z == 0):
        s = int(k * np.ceil(p * (A.shape[0] - np.count_nonzero(Z))) / (A.shape[0] - np.count_nonzero(Z)))
    else:
        s = int(k * np.ceil(p * (A.shape[0])) / (A.shape[0]))

    for i in range(1, k + 1):
        if i == (np.ceil(t_s * k) + 1):
            s = int(k * np.ceil(p * (A.shape[0])) / (A.shape[0]))

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

        size_ = ((A.shape[0] - np.count_nonzero(Z)) if i <= np.ceil(t_s * k) else A.shape[0])
        size_ = min(size_, np.count_nonzero((S + Z * (1 if i <= np.ceil(t_s * k) else 0)) == 0))
        if p < 1:
            indices = utils.random_sample_set(utils.setdiff1d_nb_faster(full_indices,
                                                                        np.nonzero(S + Z * (1 if i <= np.ceil(t_s * k)
                                                                                            else 0))[0]),
                                              int(np.ceil(p * size_)))
        else:
            indices = utils.setdiff1d_nb_faster(full_indices, np.nonzero(S + Z * (1 if i <= np.ceil(t_s * k)
                                                                                  else 0))[0])
        marginals = utils.computeMarginalGain(A=A, indices=indices, S=S, Lambda=Lambda, D=D,
                                              function_type=function_type)
        oracle_counter += indices.shape[0]

        chosen = utils.random_sample_set(np.argsort(-(marginals[indices] - base_term))[:s], k=1)[0]

        if marginals[indices[chosen]] - base_term >= 0 or generate_full_marginals:
            S[indices[chosen]] += 1
            global_marginals[indices[chosen]] = marginals[indices[chosen]] - base_term

    return S, oracle_counter, global_marginals
