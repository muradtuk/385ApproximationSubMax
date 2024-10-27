import numpy as np
import numba
import utils
from LazierThanLazyGreedyNumba import applyLazierThanLazyGreedy
from LinearTimeInit import applyQuickSWAPNM

@numba.njit('Tuple((int64, int64, int64))(float64[:,:], int64[:], int64[:],'
            ' float64, float64[:], int64, int64, float64, int64)', fastmath=True,
            parallel=True)
def evaluateAdditionOrRemoval(A, indices, S, Lambda, D, function_type, action, base_value, oracle_counter):
    marginals = np.zeros((A.shape[0],))
    e1 = np.zeros((A.shape[0],), dtype=S.dtype)

    for i in numba.prange(indices.shape[0]):
        oracle_counter += 1
        if function_type == 0:
            marginals[indices[i]] = utils.computeLocationSummarization(A=A,
                                                                       S=S + utils.numbaRoll(e1,
                                                                                             indices[i] if action > 0
                                                                                             else -1, indices[i] if
                                                                                             action < 0 else -1), D=D)
        elif function_type == 1:
            marginals[indices[i]] = utils.computeRevenueMaximizationCost(A=A,
                                                                         S=S + utils.numbaRoll(e1,
                                                                                               indices[i] if action > 0
                                                                                               else -1, indices[i] if
                                                                                               action < 0 else -1))
        elif function_type == 2:
            marginals[indices[i]] = utils.computeImageSummerizationCost(A=A,
                                                                        S=S + utils.numbaRoll(e1,
                                                                                              indices[i] if action > 0
                                                                                              else -1, indices[i] if
                                                                                              action < 0 else -1))
        elif function_type == 3:
            marginals[indices[i]] = utils.computeMovieRecommendationCost(A=A,
                                                                         S=S + utils.numbaRoll(e1,
                                                                                               indices[i] if action > 0
                                                                                               else -1, indices[i] if
                                                                                               action < 0 else -1),
                                                                         Lambda=Lambda, val=marginals[indices[i]])
    if action > 0:
        if np.max(marginals[indices] - base_value) > 0:
            return (indices[np.argmax(marginals[indices] - base_value)], oracle_counter,
                    np.max(marginals[indices] - base_value))
    else:
        return (indices[np.argmax(marginals[indices] - base_value)], oracle_counter,
                np.max(marginals[indices] - base_value))

    return -1, oracle_counter, -1


@numba.njit('Tuple((int64[:], int64))(float64[:,:], int64[:], int64, int64, float64[:], float64, float64)',
            fastmath=True)
def applyBaseFastLocalSearch(A, S_init, k, function_type, D, Lambda, epsilon):
    S = S_init
    oracle_counter = 0
    indices = np.arange(A.shape[0]).astype(np.int64)
    change = True
    # with numba.objmode(start='f8'):
    #     start = time()
    for i in range(int(8 * k / epsilon)):
        base_term = 0
        Z = utils.random_sample_set(utils.setdiff1d_nb_faster(indices, np.nonzero(S)[0]), int(A.shape[0] / k))

        if function_type == 0:
            base_term = utils.computeLocationSummarization(A=A, S=S, D=D)
        elif function_type == 1:
            base_term = utils.computeRevenueMaximizationCost(A=A, S=S)
        elif function_type == 2:
            base_term = utils.computeImageSummerizationCost(A=A, S=S)
        elif function_type == 3:
            base_term = utils.computeMovieRecommendationCost(A=A, S=S, Lambda=Lambda, val=base_term)
        else:
            raise ValueError('Please implement your own function')

        oracle_counter += 1

        u, oracle_counter, _ = evaluateAdditionOrRemoval(A=A, indices=Z, S=S, function_type=function_type,
                                                         D=D, Lambda=Lambda, action=1,
                                                         oracle_counter=oracle_counter,
                                                         base_value=base_term)
        if change:
            v, oracle_counter, _ = evaluateAdditionOrRemoval(A=A, indices=np.nonzero(S)[0], S=S,
                                                             function_type=function_type,
                                                             D=D, Lambda=Lambda, action=-1,
                                                             oracle_counter=oracle_counter,
                                                             base_value=base_term)
        new_value = 0
        if function_type == 0:
            new_value = utils.computeLocationSummarization(A=A, S=S + utils.numbaRoll(indices, u, v), D=D)
        elif function_type == 1:
            new_value = utils.computeRevenueMaximizationCost(A=A, S=S + utils.numbaRoll(indices, u, v))
        elif function_type == 2:
            new_value = utils.computeImageSummerizationCost(A=A, S=S + utils.numbaRoll(indices, u, v))
        elif function_type == 3:
            new_value = utils.computeMovieRecommendationCost(A=A, S=S + utils.numbaRoll(indices, u, v), Lambda=Lambda,
                                                             val=new_value)

        oracle_counter += 1

        if new_value >= base_term:
            S = S + utils.numbaRoll(indices, u, v)
            change = True
        else:
            change = False
    return S, oracle_counter


@numba.njit('Tuple((boolean, int64))(float64[:,:], int64[:], int64, float64, float64[:], int64, float64)',
            fastmath=True)
def checkIfGoodSet(A, S, oracle_counter, Lambda, D, function_type, epsilon):
    base_term = 0
    if function_type == 0:
        base_term = utils.computeLocationSummarization(A=A, S=S, D=D)
    elif function_type == 1:
        base_term = utils.computeRevenueMaximizationCost(A=A, S=S)
    elif function_type == 2:
        base_term = utils.computeImageSummerizationCost(A=A, S=S)
    elif function_type == 3:
        base_term = utils.computeMovieRecommendationCost(A=A, S=S, Lambda=Lambda, val=base_term)

    _, oracle_counter, added_term = evaluateAdditionOrRemoval(A=A, indices=np.nonzero(S == 0)[0], S=S,
                                                              function_type=function_type,
                                                              D=D, Lambda=Lambda, action=1,
                                                              oracle_counter=oracle_counter,
                                                              base_value=base_term)

    _, oracle_counter, removed_term = evaluateAdditionOrRemoval(A=A, indices=np.nonzero(S)[0], S=S,
                                                                function_type=function_type,
                                                                D=D, Lambda=Lambda, action=-1,
                                                                oracle_counter=oracle_counter,
                                                                base_value=base_term)

    if -(1 + epsilon) * removed_term >= added_term:
        return True, oracle_counter
    else:
        return False, oracle_counter


# @numba.njit('Tuple((int64[:], int64))(float64[:,:], int64, int64, float64[:], float64, float64,'
#             ' int64[:], float64, boolean, int64)', fastmath=True)
def applyFastLocalSearch(A, k, function_type, D, Lambda, epsilon, Z, t_s, faster, us_lazy_init):
    oracle_counter = 0
    best_S_init = None
    Z = np.zeros((A.shape[0],), dtype=np.int64)
    if us_lazy_init == 1:
        for i in range(3):  # np.ceil(np.log(10))
            (S, oracle_counter_) = applyLazierThanLazyGreedy(A=A, Z=Z, k=k, function_type=function_type, D=D,
                                                             Lambda=Lambda, epsilon=epsilon, t_s=-1, faster=faster,
                                                             generate_full_marginals=False, us_lazy_init=us_lazy_init)[:2]
            oracle_counter += oracle_counter_
            if best_S_init is None:
                best_S_init = S
            else:
                if function_type == 0:
                    val = (utils.computeLocationSummarization(A, S, D) -
                           utils.computeLocationSummarization(A, best_S_init, D))
                elif function_type == 1:
                    val = (utils.computeRevenueMaximizationCost(A, S) -
                           utils.computeRevenueMaximizationCost(A, best_S_init))
                elif function_type == 2:
                    val = (utils.computeImageSummerizationCost(A, S) -
                           utils.computeImageSummerizationCost(A, best_S_init))
                elif function_type == 3:
                    val = (utils.computeMovieRecommendationCost(A, S, Lambda, 0)
                           - utils.computeMovieRecommendationCost(A, best_S_init, Lambda, 0))
                else:
                    raise ValueError('Please implement your own function')

                oracle_counter += 2
                if val > 0:
                    best_S_init = S
    else:
        best_S_init, oracle_counter, _ = applyQuickSWAPNM(A=A, k=k, function_type=function_type, D=D, Lambda=Lambda,
                                                          beta=1)

    for i in range(int(np.ceil(np.log(1 / epsilon)))):
        S, oracle_counter_ = applyBaseFastLocalSearch(A=A, S_init=best_S_init, k=k, function_type=function_type, D=D,
                                                      Lambda=Lambda, epsilon=epsilon)

        oracle_counter += oracle_counter
        is_good, oracle_counter = checkIfGoodSet(A=A, S=S, oracle_counter=oracle_counter,
                                                 Lambda=Lambda, D=D,
                                                 function_type=function_type, epsilon=epsilon)
        if is_good:
            print(f'Found a good solution at iteration {i}')
            return S, oracle_counter
    print('No good solution found!')
    return S, oracle_counter
