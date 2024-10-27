import numpy as np
from fastargs.decorators import param
import os
from fastargs.validation import Checker, ABC
import pandas as pd
import glob
from sklearn.metrics import pairwise_distances
import plotly.graph_objs as go
import numba
import random
import torch
import torchvision
from tinyimagenet import TinyImageNet
from torch.utils.data import random_split, BatchSampler, SequentialSampler
from transformers import ViTFeatureExtractor, ViTModel
from sentence_transformers import SentenceTransformer, util
import pickle
import time
from sklearn.metrics import euclidean_distances



mathjax_config = '''
MathJax.Hub.Config({
  extensions: [],
  jax: ["input/TeX","output/HTML-CSS"],
  tex2jax: {
    inlineMath: [["$","$"],["\\(","\\)"]],
    displayMath: [["$$","$$"],["\\[","\\]"]],
    processEscapes: true
  },
  "HTML-CSS": { availableFonts: ["TeX"] }
});
'''




SUBMODULAR_PROBLEMS = {
    'Location summarization': 0,
    'Revenue maximization': 1,
    'Image summarization': 2,
    'Movie recommendation': 3
}

COLORS = {
    'Uniform sampling': (0, 0, 0),
    'Buchbinder et al. 2014': (255, 0, 0),
    'Buchbinder et al. 2017': (255, 165, 0),
    'Our algorithm slow': (0, 0, 255),
    'Algorithm 3': (0, 128, 0),
}

MARKERS = {
    'Uniform sampling': 'circle',
    'Buchbinder et al. 2014': 'diamond',
    'Buchbinder et al. 2017': 'square',
    'Our algorithm slow': 'triangle-up',
    'Algorithm 3': 'star',
}


########################################################################################################################
################################################# Taken from decile/cords ##############################################
def compute_vit_image_embeddings(images, device, return_tensor=False):
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-large-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k")
    model = model.to(device)
    sampler = BatchSampler(SequentialSampler(range(len(images))),
                           20,
                           drop_last=False)

    inputs = []
    for indices in sampler:
        if images[0].mode == 'L':
            images_batch = [images[x].convert('RGB') for x in indices]
        else:
            images_batch = [images[x] for x in indices]
        inputs.append(feature_extractor(images_batch, return_tensors="pt"))

    img_features = []
    for batch_inputs in inputs:
        tmp_feat_dict = {}
        for key in batch_inputs.keys():
            tmp_feat_dict[key] = batch_inputs[key].to(device=device)
        with torch.no_grad():
            batch_outputs = model(**tmp_feat_dict)
        batch_img_features = batch_outputs.last_hidden_state.mean(dim=1).cpu()
        img_features.append(batch_img_features)
        del tmp_feat_dict

    img_features = torch.cat(img_features, dim=0)
    if return_tensor == False:
        return img_features.numpy()
    else:
        return img_features


def compute_vit_cls_image_embeddings(images, device, return_tensor=False):
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-large-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k")
    model = model.to(device)
    sampler = BatchSampler(SequentialSampler(range(len(images))),
                           20,
                           drop_last=False)

    inputs = []
    for indices in sampler:
        if images[0].mode == 'L':
            images_batch = [images[x].convert('RGB') for x in indices]
        else:
            images_batch = [images[x] for x in indices]
        inputs.append(feature_extractor(images_batch, return_tensors="pt"))

    img_features = []
    for batch_inputs in inputs:
        tmp_feat_dict = {}
        for key in batch_inputs.keys():
            tmp_feat_dict[key] = batch_inputs[key].to(device=device)
        with torch.no_grad():
            batch_outputs = model(**tmp_feat_dict)
        batch_img_features = batch_outputs.last_hidden_state[:, 0, :].cpu()
        img_features.append(batch_img_features)
        del tmp_feat_dict

    img_features = torch.cat(img_features, dim=0)
    if return_tensor == False:
        return img_features.numpy()
    else:
        return img_features


def compute_dino_image_embeddings(images, device, return_tensor=False):
    feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vitb16')
    model = ViTModel.from_pretrained('facebook/dino-vitb16')
    model = model.to(device)
    # inputs = feature_extractor(images, return_tensors="pt")
    sampler = BatchSampler(SequentialSampler(range(len(images))),
                           20,
                           drop_last=False)
    inputs = []
    for indices in sampler:
        if images[0].mode == 'L':
            images_batch = [images[x].convert('RGB') for x in indices]
        else:
            images_batch = [images[x] for x in indices]
        inputs.append(feature_extractor(images_batch, return_tensors="pt"))

    img_features = []
    for batch_inputs in inputs:
        tmp_feat_dict = {}
        for key in batch_inputs.keys():
            tmp_feat_dict[key] = batch_inputs[key].to(device=device)
        with torch.no_grad():
            batch_outputs = model(**tmp_feat_dict)
        batch_img_features = batch_outputs.last_hidden_state.mean(dim=1).cpu()
        img_features.append(batch_img_features)
        del tmp_feat_dict

    img_features = torch.cat(img_features, dim=0)
    if return_tensor == False:
        return img_features.numpy()
    else:
        return img_features


def compute_dino_cls_image_embeddings(images, device, return_tensor=False):
    feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vitb16')
    model = ViTModel.from_pretrained('facebook/dino-vitb16')
    model = model.to(device)
    sampler = BatchSampler(SequentialSampler(range(len(images))),
                           20,
                           drop_last=False)
    inputs = []
    count = 0
    for indices in sampler:
        if images[0].mode == 'L':
            images_batch = [images[x].convert('RGB') for x in indices]
            count += 1
        else:
            images_batch = [images[x] for x in indices]
            count += 1
        inputs.append(feature_extractor(images_batch, return_tensors="pt"))

    img_features = []
    for batch_inputs in inputs:
        tmp_feat_dict = {}
        for key in batch_inputs.keys():
            tmp_feat_dict[key] = batch_inputs[key].to(device=device)
        with torch.no_grad():
            batch_outputs = model(**tmp_feat_dict)
        batch_img_features = batch_outputs.last_hidden_state[:, 0, :].cpu()
        img_features.append(batch_img_features)
        del tmp_feat_dict

    img_features = torch.cat(img_features, dim=0)
    if return_tensor == False:
        return img_features.numpy()
    else:
        return img_features


def compute_image_embeddings(model_name, images, device, return_tensor=False):
    """
    Compute image embeddings using CLIP based model and return in numpy or tensor format
    """
    model = SentenceTransformer(model_name, device=device)
    if return_tensor:
        embeddings = model.encode(images, device=device, convert_to_tensor=True).cpu()
    else:
        embeddings = model.encode(images, device=device, convert_to_numpy=True)
    return embeddings


def store_embeddings(pickle_name, embeddings, train_labels):
    """
    Store embeddings to disc
    """
    with open(pickle_name, "wb") as fOut:
        pickle.dump({'embeddings': embeddings, 'train_labels': train_labels},
                    fOut, protocol=pickle.HIGHEST_PROTOCOL)


def load_embeddings(pickle_name):
    """
    Load embeddings from disc
    """
    with open(pickle_name, "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_embeddings = stored_data['embeddings']
        try:
            train_labels = stored_data['train_labels']
        except:
            train_labels = None
    return stored_embeddings, train_labels


def dict2pickle(file_name, dict_object):
    """
    Store dictionary to pickle file
    """
    with open(file_name, "wb") as fOut:
        pickle.dump(dict_object, fOut, protocol=pickle.HIGHEST_PROTOCOL)


def pickle2dict(file_name, key):
    """
    Load dictionary from pickle file
    """
    with open(file_name, "rb") as fIn:
        stored_data = pickle.load(fIn)
        value = stored_data[key]
    return value


def get_cdist(V):
    dist_mat = euclidean_distances(V)
    return get_square(dist_mat)


@numba.jit(nopython=True, parallel=True)
def get_square(mat):
    return mat**2


def get_dot_product(mat):
    sim = np.matmul(mat, np.transpose(mat))
    return sim


########################################################################################################################
############################## Numba based implementation of our supported submodular functions ########################
@numba.jit('float64(float64[:,:], int64[:], float64, float64)', nopython=True, fastmath=True,
           parallel=True)
def computeMovieRecommendationCost(A, S, Lambda, val):
    rows = np.nonzero(S)[0]
    if len(rows) == 0:
        return 0

    for i in numba.prange(A.shape[0]):
        for j in numba.prange(rows.shape[0]):
            val += A[i, rows[j]]

    for i in numba.prange(rows.shape[0]):
        for j in numba.prange(rows.shape[0]):
            val -= Lambda * A[rows[i], rows[j]]

    return val


@numba.jit('float64(float64[:,:], int64[:], float64[:])', nopython=True, fastmath=True)
def computeLocationSummarization(A, S, D):
    val = 0
    rows = np.nonzero(S)[0]

    if len(rows) == 0:
        return 0

    for i in range(A.shape[0]):
        max_val = 0
        for j in range(rows.shape[0]):
            if max_val < A[i, rows[j]]:
                max_val = A[i, rows[j]]
        val = val + 1 / A.shape[0] * max_val

    for j in range(rows.shape[0]):
        val = val - D[rows[j]]

    return val


@numba.jit('float64(float64[:,:], int64[:])', nopython=True, fastmath=True)
def computeRevenueMaximizationCostOLD(A, S):
    rows = np.nonzero(S)[0]
    cols = np.nonzero(S == 0)[0]
    val = 0

    if len(rows) == 0:
        return 0

    for i in range(rows.shape[0]):
        for j in range(cols.shape[0]):
            val = val + A[rows[i], cols[j]]

    return val


@numba.jit('float64(float64[:,:], int64[:])', nopython=True, fastmath=True, parallel=True)
def computeRevenueMaximizationCost2(A, S):
    rows = np.nonzero(S)[0]
    cols = np.nonzero(S == 0)[0]

    if len(rows) == 0:
        return 0
    
    # Initialize a global sum for final result
    total_val = 0.0

    # Outer loop parallelized
    for i in numba.prange(rows.shape[0]):
        # Inner loop parallelized as well, with thread-local variable for accumulation
        local_val = 0.0
        for j in numba.prange(cols.shape[0]):  # Both loops parallelized
            local_val += A[rows[i], cols[j]]  # Accumulate locally within thread
        # Safely accumulate the local_val from each thread
        total_val += local_val

    return total_val


@numba.njit('float64(float64[:,:], int64[:])', parallel=True, fastmath=True)
def computeRevenueMaximizationCost(A, S):
    rows = np.nonzero(S)[0]
    cols = np.nonzero(S == 0)[0]
    
    if len(rows) == 0:
        return 0.0

    # Initialize a global sum for final result
    total_val = 0.0

    # Precompute lengths to reduce redundant calls
    num_rows = rows.shape[0]
    num_cols = cols.shape[0]

    # Outer loop parallelized
    for i in numba.prange(num_rows):
        local_val = 0.0  # Thread-local variable for accumulation
        row_i = rows[i]  # Cache the current row index to reduce memory access
        
        # Manually unroll the inner loop while ensuring the step size is 1
        # Sequential unrolling, can't use prange inside prange with step > 1
        j = 0
        while j <= num_cols - 4:
            local_val += A[row_i, cols[j]] 
            local_val += A[row_i, cols[j+1]]
            local_val += A[row_i, cols[j+2]]
            local_val += A[row_i, cols[j+3]]
            j += 4

        # Handle remaining elements
        while j < num_cols:
            local_val += A[row_i, cols[j]]
            j += 1

        # Safely accumulate the local_val from each thread
        total_val += local_val

    return total_val



@numba.jit('float64(float64[:,:], int64[:])', nopython=True, fastmath=True)
def computeImageSummerizationCost(A, S):
    rows = np.nonzero(S)[0]
    val = 0

    if len(rows) == 0:
        return 0

    for i in range(A.shape[0]):
        max_val = 0
        for j in range(rows.shape[0]):
            if max_val < A[i, rows[j]]:
                max_val = A[i, rows[j]]
        val += max_val

    for i in range(rows.shape[0]):
        for j in range(rows.shape[0]):
            val -= A[rows[i], rows[j]] / A.shape[0]

    return val


########################################################################################################################
################################################### Utility functions ##################################################
@numba.njit('int64[:](int64[:], int64, int64)', fastmath=True, cache=True)
def numbaRoll(e1, idx0=-1, idx1=-1):
    b = np.empty_like(e1, dtype=e1.dtype)

    for i in range(e1.shape[0]):
        if i == idx0:
            b[i] = 1
        elif i == idx1:
            b[i] = -1
        else:
            b[i] = 0

    return b


@numba.njit('int64[:](int64[:], int64)', fastmath=True)
def random_sample_set(arr, k=-1):
    n = arr.size
    if k < 0:
        k = arr.size
    seen = {0}
    seen.clear()
    index = np.empty(k, dtype=np.int64)
    for i in range(k):
        j = random.randint(i, n - 1)
        while j in seen:
            j = random.randint(0, n - 1)
        seen.add(j)
        index[i] = j
    return arr[index]


@numba.njit('int64(int64)')
def hash_32bit_4k(value):
    return (np.int64(value) * np.int64(27_644_437)) & np.int64(0x0FFF)


@numba.njit(['int64[:](int64[:], int64[:])', 'int64[:](int64[::1], int64[::1])'])
def setdiff1d_nb_faster(arr1, arr2):
    out = np.empty_like(arr1)
    bloomFilter = np.zeros(4096, dtype=np.uint8)
    for j in range(arr2.size):
        bloomFilter[hash_32bit_4k(arr2[j])] = True
    cur = 0
    for i in range(arr1.size):
        # If the bloom-filter value is true, we know arr1[i] is not in arr2.
        # Otherwise, there is maybe a false positive (conflict) and we need to check to be sure.
        if bloomFilter[hash_32bit_4k(arr1[i])] and arr1[i] in arr2:
            continue
        out[cur] = arr1[i]
        cur += 1
    return out[:cur]


@numba.njit('float64[:](float64[:,:], int64[:], int64[:], float64, float64[:], int64)', fastmath=True,
            parallel=True)
def computeMarginalGain(A, indices, S, Lambda, D, function_type):
    marginals = np.zeros((A.shape[0],))
    e1 = np.zeros((A.shape[0], ), dtype=S.dtype)

    for i in numba.prange(indices.shape[0]):
        if function_type == 0:
            marginals[indices[i]] = computeLocationSummarization(A, S + numbaRoll(e1, indices[i], -1), D)
        elif function_type == 1:
            marginals[indices[i]] = computeRevenueMaximizationCost(A, S + numbaRoll(e1, indices[i], -1))
        elif function_type == 2:
            marginals[indices[i]] = computeImageSummerizationCost(A, S + numbaRoll(e1, indices[i], -1))
        elif function_type == 3:
            marginals[indices[i]] = computeMovieRecommendationCost(A=A, S=(S + numbaRoll(e1, indices[i], -1)),
                                                                   Lambda=Lambda, val=marginals[indices[i]])

    return marginals


########################################################################################################################
################################################# Fast args related utilities ##########################################
class PositiveInteger(Checker, ABC):
    def __init__(self, low=1):
        self.low = low

    def check(self, value):
        if int(value) < self.low:
            raise ValueError()
        return int(value)

    def help(self):
        return "Should satisfy positivity constraints"


def retrieveDelimeter(file_path):
    reader = pd.read_csv(file_path, sep=None, iterator=True)
    inferred_sep = reader._engine.data.dialect.delimiter
    return inferred_sep


@param('Submodular.submodular_problem')
def obtainRangeOfKs(submodular_problem):
    if 'image' in submodular_problem.lower():
        return np.linspace(2, 11, num=10).astype(int)
    else:
        return np.linspace(10, 100, num=10).astype(int)


@param('Dataset.dataset_name')
@param('Dataset.similarity_type')
@param('Submodular.epsilon')
@param('Submodular.submodular_problem')
@param('Embedding.model')
def readData(dataset_name, similarity_type, epsilon, submodular_problem, model, return_labels=False, seed=42,
             traditional=False):
    repeat_ = 0
    train_labels = None
    D = None
    if not os.path.isdir(f'Data/{dataset_name}/'):
        raise ValueError('Please create a dataset directory inside Data directory by your desired name '
                         'from which the data will be pulled from.')
    else:
        if 'cifar' not in dataset_name.lower() and 'imagenet' not in dataset_name.lower():
            Lambda = 0.55
            files = list(glob.glob(f'Data/{dataset_name}/*'))
            if any([f'similarity_matrix_{similarity_type}' in file for file in files]):
                matching = [file for file in files if f'similarity_matrix_{similarity_type}' in file][0]
                temp = np.load(matching, allow_pickle=True)
                if 'D' in temp.keys():
                    D = temp['D']
                else:
                    D = None

                S = temp['S'].astype(np.float64)
                if D is None:
                    D = np.zeros((S.shape[0], ))

            else:
                for file_path in files:
                    if (file_path.endswith('.txt') or file_path.endswith('.csv') or file_path.endswith('.xslx') or
                            file_path.endswith('.anon')):
                        if 'yelp' in dataset_name.lower():
                            if 'sim' not in file_path:
                                continue
                        delimeter = retrieveDelimeter(file_path)
                        P = pd.read_csv(file_path, header=None, sep=delimeter)

                        if any([x in file_path.lower() for x in ['facebook', 'edges', 'advogato']]):
                            if 'facebook' in file_path.lower():
                                P = P.iloc[1:].to_numpy()
                                P[P[:, -1] != r'\N', -1] = 1
                                P[P[:, -1] == r'\N', -1] = 0
                            elif 'edges' in file_path.lower() or 'advogato' in file_path.lower():
                                P = P.iloc[1:].to_numpy().astype('float')

                            max_index = P[:, :-1].max()
                            min_index = P[:, :-1].min()
                            remove = int(min_index)
                            n = int(max_index - min_index + 1)

                            S = np.zeros((n, n))

                            S[P[:, 0].astype('int') - remove, P[:, 1].astype('int') - remove] += P[:, -1].astype(int)
                            D = np.zeros((S.shape[0],))
                            np.savez(f'Data/{dataset_name}/similarity_matrix_{similarity_type}.npz', S=S, D=D)
                        elif 'yelp' in file_path.lower():
                            dists = file_path.replace('sim', 'people')
                            D = pd.read_csv(dists, header=None, sep=delimeter).to_numpy().astype(np.float64)
                            D = np.sum(D, axis=0) / 100_000
                            S = P.values.astype(np.float64)
                            np.savez(f'Data/{dataset_name}/similarity_matrix_{similarity_type}.npz', S=S, D=D)
                        else:
                            if similarity_type == 'graph':
                                raise ValueError('Graph type similarity is only availble for Yelp dataset,'
                                                 ' Facebook dataset, and Advogato dataset')
                            elif similarity_type == 'dot_product':
                                S = P.values.dot(P.values.T)
                            else:
                                S = pairwise_distances(P.values, metric=similarity_type)

                            D = np.zeros((S.shape[0],))
                            np.savez(f'Data/{dataset_name}/similarity_matrix_{similarity_type}.npz', S=S, D=D)
                        break
        else:
            if not os.path.exists(f'Data/{dataset_name.upper()}/{dataset_name.lower()}_{model}_train_embeddings.pkl'):
                if 'cifar10' == dataset_name.lower():
                    fullset = torchvision.datasets.CIFAR10(root=f'Data/{dataset_name.upper()}', train=True,
                                                           download=True, transform=None)
                    validation_set_fraction = 0.1
                    num_fulltrn = len(fullset)
                    num_val = int(num_fulltrn * validation_set_fraction)
                    num_trn = num_fulltrn - num_val
                    trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))
                    repeat_ = 34
                elif dataset_name.lower() == 'cifar100':
                    fullset = torchvision.datasets.CIFAR100(root=f'Data/{dataset_name.upper()}', train=True,
                                                            download=True, transform=None)

                    validation_set_fraction = 0.1
                    num_fulltrn = len(fullset)
                    num_val = int(num_fulltrn * validation_set_fraction)
                    num_trn = num_fulltrn - num_val
                    trainset, valset = random_split(fullset, [num_trn, num_val],
                                                    generator=torch.Generator().manual_seed(seed))
                    repeat_ = 34
                elif dataset_name.lower() == 'tinyimagenet':
                    fullset = TinyImageNet(root=f'Data/{dataset_name.upper()}', split='train', download=True, transform=None)
                    testset = TinyImageNet(root=f'Data/{dataset_name.upper()}', split='val', download=True, transform=None)
                    validation_set_fraction = 0.1
                    num_fulltrn = len(fullset)
                    num_val = int(num_fulltrn * validation_set_fraction)
                    num_trn = num_fulltrn - num_val
                    trainset, valset = random_split(fullset, [num_trn, num_val],
                                                    generator=torch.Generator().manual_seed(seed))
                    repeat_ = 15
                else:
                    raise NotImplementedError('Please implement your dataset loader')

                device = 'cuda:0'

                train_images = [x[0] for x in trainset]
                train_labels = [x[1] for x in trainset]

                if model == 'ViT':
                    S = compute_vit_image_embeddings(train_images, device)
                elif model == 'ViT_cls':
                    S = compute_vit_cls_image_embeddings(train_images, device)
                elif model == 'dino':
                    S = compute_dino_image_embeddings(train_images, device)
                elif model == 'dino_cls':
                    S = compute_dino_cls_image_embeddings(train_images, device)
                else:
                    S = compute_image_embeddings(model, train_images, device)
                store_embeddings(f'Data/{dataset_name.upper()}/' + dataset_name.lower()
                                 + '_' + model + '_train_embeddings.pkl', S,
                                 train_labels=[x[1] for x in trainset])
                D = np.zeros((S.shape[0],), dtype=np.float64)
            else:
                S, train_labels = load_embeddings(f'Data/{dataset_name.upper()}/' + dataset_name.lower()
                                    + '_' + model + '_train_embeddings.pkl')
                D = np.zeros((S.shape[0], ), dtype=np.float64)

                if traditional:
                    if os.path.exists(f'Data/{dataset_name.upper()}/similarity_matrix_{similarity_type}.npz'):
                        temp = np.load(f'Data/{dataset_name.upper()}/similarity_matrix_{similarity_type}.npz', allow_pickle=True)
                        if 'D' in temp.keys():
                            D = temp['D']
                        else:
                            D = None

                        S = temp['S'].astype(np.float64)
                        if D is None:
                            D = np.zeros((S.shape[0],), dtype=np.float64)
                        Lambda = 0.55
                    else:
                        if S.shape[0] > 10_000:
                            indices = np.random.choice(S.shape[0], size=10_000, replace=False)
                            S = S[indices]
                            S = pairwise_distances(S, metric=similarity_type).astype(np.float64)
                            D = np.zeros((S.shape[0],), dtype=np.float64)
                            np.savez(f'Data/{dataset_name.upper()}/similarity_matrix_{similarity_type}.npz', S=S,
                                     D=D)
                            Lambda = 0.75
                else:
                    repeat_ = 34 if 'cifar' in dataset_name.lower() else 15
                    Lambda = 0.35

    if not return_labels:
        return S, Lambda, epsilon, SUBMODULAR_PROBLEMS[submodular_problem], D
    else:
        return S, Lambda, epsilon, SUBMODULAR_PROBLEMS[submodular_problem], D, train_labels, repeat_


def rgb_to_rgba(rgb_value, alpha):
    """
    Adds the alpha channel to an RGB Value and returns it as an RGBA Value
    :param rgb_value: Input RGB Value
    :param alpha: Alpha Value to add  in range [0,1]
    :return: RGBA Value
    """
    return f"rgba{rgb_value[:] + (alpha,)}"


def to_rgb(rgb_value):
    return f"rgb{rgb_value[:]}"

@param('Experimentation.save_path')
@param('Dataset.dataset_name')
@param('Submodular.submodular_problem')
@param('Experimentation.graph_plotting')
@param('Dataset.similarity_type')
@param('Experimentation.alpha_opacity')
@param('Experimentation.as_function_of_lambda')
@param('Experimentation.graph_compare')
def save_experiment(Ks, arrays, legends, save_only, load_results, **kwargs):
    # save npz arrays
    path = kwargs['save_path'] + '/' + kwargs['dataset_name'] + '/' + kwargs['submodular_problem'] + '/' + kwargs[
        'similarity_type']
    if not load_results:
        os.makedirs(path, exist_ok=True)
        np.savez(path + '/results_summary.npz', function_values=arrays[0],
                 oracle_calls=arrays[1], legends=legends)
        function_values = arrays[0]
        oracle_calls = arrays[1]
    else:
        temp = np.load(path + '/results_summary.npz', allow_pickle=True)
        function_values = temp['function_values']#[:, :5, :]
        oracle_calls = temp['oracle_calls']#[:, :5, :]
        Ks = Ks if not kwargs['as_function_of_lambda'] and np.isscalar(kwargs['Lambdas']) else kwargs['Lambdas']
        Ks = Ks#[:5]
        legends = temp['legends'].tolist()
        legends = [x.replace('et.', 'et') for x in legends]

        legends[-1] = 'Algorithm 3'

        if kwargs['graph_compare']:
            temp2 = np.load(path + '2/results_summary.npz', allow_pickle=True)
            function_values2 = temp2['function_values']
            oracle_calls2 = temp2['oracle_calls']

        legends = [
            x if x in COLORS.keys() else ('Buchbinder et al. 2017' if 'lazy' in x else 'Buchbinder et al. 2014') for x
            in legends]

    if kwargs['graph_plotting'] and not save_only:
        if len(Ks) > 1 and function_values.shape[0] > 1:
            fig_func_val = go.Figure()
            fig_oracle_cnt = go.Figure()
            y_min = y_min_oracle = np.inf
            y_max = y_max_oracle = -np.inf

            for i in range(function_values.shape[0]):
                if i == (function_values.shape[0] - 1) and kwargs['graph_plotting'] and kwargs['graph_compare']:
                    suffix = ' with different initialization'
                else:
                    suffix = ''

                if i == 0:
                    continue
                if i != (function_values.shape[0] -1) or True:
                    mean_val = np.mean(function_values[i], axis=1)
                    if y_min > mean_val.min():
                        y_min = mean_val.min()
                    if y_max < mean_val.max():
                        y_max = mean_val.max()
                    mean_oracle = np.mean(oracle_calls[i], axis=1)
                    std_val = np.std(function_values[i], axis=1)
                    std_oracle = np.std(oracle_calls[i], axis=1)

                    if y_min_oracle > mean_oracle.min():
                        y_min_oracle = mean_oracle.min()
                    if y_max_oracle < mean_oracle.max():
                        y_max_oracle = mean_oracle.max()

                    if i == (function_values.shape[0] -1) and kwargs['graph_plotting'] and kwargs['graph_compare']:
                        mean_val2 = np.mean(function_values2[i], axis=1)
                        mean_oracle2 = np.mean(oracle_calls2[i], axis=1)
                        std_val2 = np.std(function_values2[i], axis=1)
                        std_oracle2 = np.std(oracle_calls2[i], axis=1)

                fig_func_val.add_trace(go.Scatter(
                    x=Ks,
                    y=mean_val,
                    name=legends[i] + suffix,
                    mode='markers+lines',
                    marker=dict(
                        color=to_rgb(COLORS[legends[i]]),
                        line_width=1,
                        symbol=MARKERS[legends[i]]
                    ),
                    line=dict(color=to_rgb(COLORS[legends[i]]))
                ))

                if i == (function_values.shape[0] - 1) and kwargs['graph_plotting'] and kwargs['graph_compare']:
                    fig_func_val.add_trace(go.Scatter(
                        x=Ks,
                        y=mean_val2,
                        name=legends[i],
                        mode='markers+lines',
                        marker=dict(
                            color=to_rgb(COLORS[legends[i]]),
                            line_width=1,
                            symbol=MARKERS[legends[i]]
                        ),
                        line=dict(color=to_rgb(COLORS[legends[i]]), dash='dot')
                    ))
                fig_func_val.add_trace(go.Scatter(
                    x=Ks.tolist() + Ks.tolist()[::-1],  # x, then x reversed
                    y=(mean_val + std_val).tolist() + (mean_val - std_val).tolist()[::-1],  # upper, then lower reversed
                    fill='toself',
                    fillcolor=rgb_to_rgba(COLORS[legends[i]], kwargs['alpha_opacity']),
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                )
                )

                if i == (function_values.shape[0] - 1) and kwargs['graph_plotting'] and kwargs['graph_compare']:
                    fig_func_val.add_trace(go.Scatter(
                        x=Ks.tolist() + Ks.tolist()[::-1],  # x, then x reversed
                        y=(mean_val2 + std_val2).tolist() + (mean_val2 - std_val2).tolist()[::-1],
                        # upper, then lower reversed
                        fill='toself',
                        fillcolor=rgb_to_rgba(COLORS[legends[i]], kwargs['alpha_opacity']),
                        line=dict(color='rgba(255,255,255,0)', dash='dot'),
                        hoverinfo="skip",
                        showlegend=False
                    )
                    )

                fig_oracle_cnt.add_trace(go.Scatter(
                    x=Ks,
                    y=mean_oracle,
                    name=legends[i] + suffix,
                    mode='markers+lines',
                    marker=dict(
                        color=to_rgb(COLORS[legends[i]]),
                        line_width=1,
                        symbol=MARKERS[legends[i]]
                    ),
                    line=dict(color=to_rgb(COLORS[legends[i]]))
                ))
                if i == (function_values.shape[0] - 1) and kwargs['graph_plotting'] and kwargs['graph_compare']:
                    fig_oracle_cnt.add_trace(go.Scatter(
                        x=Ks,
                        y=mean_oracle2,
                        name=legends[i],
                        mode='markers+lines',
                        marker=dict(
                            color=to_rgb(COLORS[legends[i]]),
                            line_width=1,
                            symbol=MARKERS[legends[i]]
                        ),
                        line=dict(color=to_rgb(COLORS[legends[i]]), dash='dot')
                    ))

                fig_oracle_cnt.add_trace(go.Scatter(
                    x=Ks.tolist() + Ks.tolist()[::-1],  # x, then x reversed
                    y=(mean_oracle + std_oracle).tolist() + (mean_oracle - std_oracle).tolist()[::-1],  # upper, then lower reversed
                    fill='toself',
                    fillcolor=rgb_to_rgba(COLORS[legends[i]], kwargs['alpha_opacity']),
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                )
                )

                if i == (function_values.shape[0] - 1) and kwargs['graph_plotting'] and kwargs['graph_compare']:
                    fig_oracle_cnt.add_trace(go.Scatter(
                        x=Ks.tolist() + Ks.tolist()[::-1],  # x, then x reversed
                        y=(mean_oracle2 + std_oracle2).tolist() + (mean_oracle2 - std_oracle2).tolist()[::-1],
                        # upper, then lower reversed
                        fill='toself',
                        fillcolor=rgb_to_rgba(COLORS[legends[i]], kwargs['alpha_opacity']),
                        line=dict(color='rgba(255,255,255,0)', dash='dot'),
                        hoverinfo="skip",
                        showlegend=False
                    )
                    )

            if not kwargs['as_function_of_lambda']:
                fig_func_val.update_layout(xaxis_title='k',
                                           yaxis_title='Function value')

                fig_oracle_cnt.update_layout(xaxis_title='k',
                                             yaxis_title='Oracle calls')
            else:
                fig_func_val.update_layout(xaxis_title=r'$\lambda$',
                                           yaxis_title='Function value')

                fig_oracle_cnt.update_layout(xaxis_title=r'$\lambda$',
                                             yaxis_title='Oracle calls')

            y_range = y_max*1.1 - y_min*0.8
            y_dtick = y_range / 5

            fig_func_val.update_layout(
                font=dict(
                    family="Courier New, monospace",
                    size=28,  # Set the font size here
                ),
                yaxis=dict(
                    tickmode='linear',  # Use linear mode for ticks
                    dtick=y_dtick,  # Set the interval between ticks for y-axis
                    range=[y_min*0.8, y_max*1.1]  # Set the range for y-axis
                ),
            )
            fig_func_val.update_layout(
                legend=dict(
                    x=0.45,
                    y=0.04,
                    traceorder="reversed",
                    bordercolor="Black",
                    borderwidth=2,
                    bgcolor='rgba(255,255,255,0)'
                ),
                width=1000,
                height=700,
            )

            y_range_oracle = y_max_oracle*1.1 - y_min_oracle*0.8
            y_dtick_oracle = y_range_oracle / 5
            fig_oracle_cnt.update_layout(
                font=dict(
                    family="Courier New, monospace",
                    size=28,  # Set the font size here
                ),
                yaxis=dict(
                    tickmode='linear',  # Use linear mode for ticks
                    dtick=y_dtick_oracle,  # Set the interval between ticks for y-axis
                    range=[y_min_oracle * 0.8, y_max_oracle * 1.1]  # Set the range for y-axis
                ),
            )

            fig_oracle_cnt.update_layout(
                legend=dict(
                    x=0.05,
                    y=0.95,
                    traceorder="reversed",
                    bordercolor="Black",
                    borderwidth=2,
                    bgcolor='rgba(255,255,255,0)'
                ),
                width=1000,
                height=700
            )

            fig_oracle_cnt.write_image(path + '/oracle.pdf')
            time.sleep(5)
            fig_oracle_cnt.write_image(path + '/oracle.png')
            time.sleep(5)
            fig_oracle_cnt.write_image(path + '/oracle.pdf')
            time.sleep(5)
            fig_func_val.write_image(path + '/function_val.pdf')
            time.sleep(5)
            fig_func_val.write_image(path + '/function_val.png')
            time.sleep(5)



@param('Experimentation.save_path')
@param('Dataset.dataset_name')
@param('Submodular.submodular_problem')
@param('Dataset.similarity_type')
def saveResultDeepLearning(S, marignals, **kwargs):
    path = kwargs['save_path'] + '/' + kwargs['dataset_name'] + '/' + kwargs['submodular_problem'] + '/' + kwargs[
        'similarity_type']

    np.savez(path + '/subsets.npz', S=S, marignals=marignals)



@param('Experimentation.save_path')
@param('Dataset.dataset_name')
@param('Submodular.submodular_problem')
@param('Dataset.similarity_type')
def savePickles(list_, key, **kwargs):
    file_name = f'cifar10_dino_cls_cossim_gc_pc_0.01_{kwargs["fraction"]}_stochastic_subsets.pkl' if ("fraction" in
                                                                                                      kwargs.keys())\
        else 'key.pkl'
    file_path = kwargs['save_path'] + '/' + kwargs['dataset_name'] + '/' + kwargs['submodular_problem'] + '/' + kwargs[
        'similarity_type'] + '/' + file_name

    dict2pickle(file_path, {key: list_})


@param('Experimentation.save_path')
@param('Dataset.dataset_name')
@param('Submodular.submodular_problem')
@param('Dataset.similarity_type')
@param('Embedding.model')
def savePicklesForMILOBasedCode(**kwargs):
    file_path = kwargs['save_path'] + '/' + kwargs['dataset_name'] + '/' + kwargs['submodular_problem'] + '/' + kwargs[
        'similarity_type'] + '/'
    dict2pickle(file_path + kwargs['dataset_name'].lower() + '_' + kwargs['model'].lower() + '_cossim_' +
                'gc_pc' + '_' + str(kwargs['kw']) + '_global_order.pkl',
                {'globalorder': kwargs['global_order'], 'cluster_idxs': kwargs['cluster_idxs']})
    dict2pickle(file_path + kwargs['dataset_name'].lower() + '_' + kwargs['model'].lower() + '_cossim_' +
                'gc_pc' + '_' + str(kwargs['r2_coefficient']) + '_global_r2.pkl',
                {'globalr2': kwargs['global_r2'], 'cluster_idxs': kwargs['cluster_idxs']})
    dict2pickle(file_path + kwargs['dataset_name'].lower() + '_' + kwargs['model'].lower() + '_cossim_' +
                'gc_pc' + '_' + str(kwargs['knn']) + '_global_knn.pkl',
                {'globalknn': kwargs['global_knn'], 'cluster_idxs': kwargs['cluster_idxs']})


@param('Experimentation.save_path')
@param('Dataset.dataset_name')
@param('Submodular.submodular_problem')
@param('Dataset.similarity_type')
def loadPickles(dict_obj, key, **kwargs):
    file_path = kwargs['save_path'] + '/' + kwargs['dataset_name'] + '/' + kwargs['submodular_problem'] + '/' + kwargs[
        'similarity_type'] + '/' + f'{key}.pkl'

    return pickle2dict(file_path, dict_obj)


@param('Experimentation.save_path')
@param('Dataset.dataset_name')
@param('Submodular.submodular_problem')
@param('Dataset.similarity_type')
def saveNumpyObj(dict_obj, key, **kwargs):
    file_path = kwargs['save_path'] + '/' + kwargs['dataset_name'] + '/' + kwargs['submodular_problem'] + '/' + kwargs[
        'similarity_type'] + '/'
    file_name = file_path + f'{key}.pkl'

    os.makedirs(file_path, exist_ok=True)
    dict2pickle(file_name, {key: dict_obj})


@param('Experimentation.save_path')
@param('Dataset.dataset_name')
@param('Submodular.submodular_problem')
@param('Dataset.similarity_type')
def loadNumpyObj(key, **kwargs):
    file_path = kwargs['save_path'] + '/' + kwargs['dataset_name'] + '/' + kwargs['submodular_problem'] + '/' + kwargs[
        'similarity_type'] + '/' + f'{key}.pkl'

    return pickle2dict(file_path, key)
