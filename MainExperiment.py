import numpy as np
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf, InRange
from argparse import ArgumentParser
import utils
from UniformSampling import applyUniformSampling
from decimal import Decimal
from tqdm import tqdm
from RandomGreedyNumba import applyRandomGreedy
from LazierThanLazyGreedyNumba import applyLazierThanLazyGreedy
from OurSolver import obtainOurSolutionFast


Section('Dataset', 'Dataset parameters').params(
    dataset_name=Param(str, 'Dataset name', default='Movies'),
    similarity_type=Param(And(str, OneOf(['cosine', 'euclidean', 'dot_product', 'graph'])), 'Similarity metric',
                          default='cosine'),
)


Section('Submodular', 'Submodular parameters').params(
    submodular_problem=Param(And(str, OneOf(['Image summarization', 'Revenue maximization',
                                             'Location summarization', 'Movie recommendation'])),
                             'Submodular problem', default='Image summarization', required=True),
    competitor=Param(And(str, OneOf(['Uniform Sampling', 'Buchbinder et. al. 2014',
                                                       'Buchbinder et. al. 2017',])),
                                       'Competitor submodular', default="Random greedy", required=True),
    K=Param(And(int, utils.PositiveInteger(low=1)), 'Cardinality constraint', default=10, required=True),
    epsilon=Param(And(float, InRange(0.0, 1.0)), 'Epsilon', default=0.1, required=True),
    us_lazy_init=Param(OneOf([0, 1]), default=1, required=True)
)


Section('Experimentation', 'Experimentation parameters').params(
    save_path=Param(str, 'Experiment path', default='Results'),
    allvall=Param(bool, 'Compare solvers against each other', default=False),
    multiple_ks=Param(bool, 'Apply multiple Ks', default=False),
    graph_plotting=Param(bool, 'Plot graph', default=False),
    load_results=Param(bool, 'Rerun experiment', default=False),
    reps=Param(And(int, utils.PositiveInteger(1)), 'Number of repetitions', default=1),
    alpha_opacity=Param(And(float, InRange(0.0, 1.0)), 'Alpha opacity for graph plotting',
                        default=0.5),
    update_results=Param(bool, 'Update results', default=False),
    as_function_of_lambda=Param(bool, 'Conduct experiments as function of Lambda', default=False),
    graph_compare=Param(bool, 'Compare', default=False, required=True)
)


Section('Embedding', 'Embedding parameters').params(
    model=Param(And(str, OneOf(['ViT', 'ViT_cls', 'dino', 'dino_cls'])), 'Embedding', default='ViT',
                required=False),
)


def make_config(quiet=False, description=''):
    config = get_current_config()
    parser = ArgumentParser(description=description)
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()
    return config


@param('Experimentation.allvall')
@param('Submodular.competitor')
@param('Submodular.K')
@param('Experimentation.multiple_ks')
@param('Experimentation.load_results')
@param('Experimentation.reps')
@param('Submodular.submodular_problem')
@param('Dataset.similarity_type')
@param('Experimentation.update_results')
@param('Experimentation.as_function_of_lambda')
@param('Submodular.us_lazy_init')
def main(**kwargs):
    if not kwargs['load_results']:
        data, Lambda, epsilon, function_type, D = utils.readData(traditional=True)
        data = data
        t_s = 0.372  # to ensure 0.385 approximation

    if kwargs['allvall']:
        algorithms = [applyUniformSampling, applyRandomGreedy, applyLazierThanLazyGreedy,
                      obtainOurSolutionFast]
        legends = ['Uniform sampling', 'Buchbinder et al. 2014', 'Buchbinder et al. 2017',
                   'Algorithm 1']

    if not kwargs['multiple_ks'] or kwargs['as_function_of_lambda']:
        Ks = [kwargs['K']]
        Lambdas = np.linspace(0.1, 1, 10).astype(np.float64)
    else:
        Ks = utils.obtainRangeOfKs()
        Lambdas = 0.55 #Lambda

    if not kwargs['update_results']:
        if np.isscalar(Lambdas):
            function_value = np.zeros((len(legends), len(Ks), kwargs['reps']))
            counter_value = np.zeros((len(legends), len(Ks), kwargs['reps']))
        else:
            function_value = np.zeros((len(legends), len(Lambdas), kwargs['reps']))
            counter_value = np.zeros((len(legends), len(Lambdas), kwargs['reps']))
    else:
        function_value = np.zeros((len(legends), len(Ks), kwargs['reps']))
        counter_value = np.zeros((len(legends), len(Ks), kwargs['reps']))

    if not kwargs['load_results']:
        for j in (range(len(Ks) if np.isscalar(Lambdas) else len(Lambdas))):
            for i in range(len(legends)):
                if kwargs['update_results'] and i == (len(legends) - 1):
                    continue
                for l in tqdm(range(kwargs['reps'])):
                    (S, counter_value[i, j, l]) = algorithms[i](A=data, k=Ks[j] if np.isscalar(Lambdas) else Ks[0],
                                                                epsilon=epsilon,
                                                                t_s=t_s if 'Algorithm' in legends[i] else -1, D=D,
                                                                Lambda=Lambda if np.isscalar(Lambdas) else Lambdas[j],
                                                                function_type=function_type,
                                                                Z=np.zeros((data.shape[0], ), dtype=np.int64),
                                                                faster=False,
                                                                generate_full_marginals=False,
                                                                us_lazy_init=kwargs['us_lazy_init'])[:2]
                    if function_type == 0:
                        function_value[i, j, l] = utils.computeLocationSummarization(A=data, D=D, S=S)
                    elif function_type == 1:
                        function_value[i, j, l] = utils.computeRevenueMaximizationCost(A=data, S=S)
                    elif function_type == 2:
                        function_value[i, j, l] = utils.computeImageSummerizationCost(A=data, S=S)
                    elif function_type == 3:
                        function_value[i, j, l] = utils.computeMovieRecommendationCost(A=data, S=S, Lambda=Lambda,
                                                                                       val=function_value[i, j, l])
                    else:
                        raise NotImplementedError('Please implement your desired submodular function')

                    print(f'method {legends[i]} for k = {Ks[j] if np.isscalar(Lambdas) else Ks[0]}'
                          rf'and $\lambda$={Lambdas if np.isscalar(Lambdas) else Lambdas[j]} got function'
                          f'value {Decimal(function_value[i,j,l]):.5E}'
                          f' while requiring {counter_value[i, j,l]} oracle calls ')
                    utils.save_experiment(Ks, arrays=[function_value, counter_value],
                                          legends=legends,load_results=False, save_only=True, Lambdas=Lambdas)
    utils.save_experiment(Ks,  arrays=[function_value, counter_value], legends=legends,
                          load_results=kwargs['load_results'], save_only=False, Lambdas=Lambdas)


if __name__ == '__main__':
    args = make_config(description='Submodular traditional experiments')
    main()