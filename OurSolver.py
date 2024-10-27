import numpy as np
import numba
from LazierThanLazyGreedyNumba import applyLazierThanLazyGreedy
from FastLocalSearchNumba import applyFastLocalSearch
import utils

# @numba.njit('Tuple((int64[:], int64, float64[:]))(float64[:,:], int64, float64[:], float64, int64,'
#             ' float64, float64, int64[:], boolean, boolean, int64)', fastmath=True)
def obtainOurSolutionFast(A, k, D, Lambda, function_type, epsilon, t_s, Z, faster=False, generate_full_marginals=False, us_lazy_init=1):
    S1, oracle_calls1 = applyFastLocalSearch(A=A, k=k, Lambda=Lambda, function_type=function_type, epsilon=epsilon,
                                             D=D, Z=Z, t_s=t_s, faster=faster, us_lazy_init=us_lazy_init)

    S2, oracle_calls2, full_marginals = applyLazierThanLazyGreedy(A=A, Z=S1, k=k, function_type=function_type,
                                                                  epsilon=epsilon, D=D,
                                                                  t_s=t_s, Lambda=Lambda, faster=False,
                                                                  generate_full_marginals=generate_full_marginals,
                                                                  us_lazy_init=us_lazy_init)
    base_term1 = 0
    base_term2 = 0
    if function_type == 0:
        base_term1 = utils.computeLocationSummarization(A=A, S=S1, D=D)
        base_term2 = utils.computeLocationSummarization(A=A, S=S2, D=D)
    elif function_type == 1:
        base_term1 = utils.computeRevenueMaximizationCost(A=A, S=S1)
        base_term2 = utils.computeRevenueMaximizationCost(A=A, S=S2)
    elif function_type == 2:
        base_term1 = utils.computeImageSummerizationCost(A=A, S=S1)
        base_term2 = utils.computeImageSummerizationCost(A=A, S=S2)
    elif function_type == 3:
        base_term1 = utils.computeMovieRecommendationCost(A=A, S=S1, Lambda=Lambda, val=base_term1)
        base_term2 = utils.computeMovieRecommendationCost(A=A, S=S2, Lambda=Lambda, val=base_term2)
    else:
        raise ValueError("Please implement your desired function")

    if base_term1 > base_term2 and not generate_full_marginals:
        return S1, oracle_calls1 + oracle_calls2, np.zeros((S2.shape[0], ))
    else:
        if not generate_full_marginals:
            return S2, oracle_calls2 + oracle_calls1, np.zeros((S2.shape[0], ))
        else:
            return S2, oracle_calls2 + oracle_calls1, full_marginals