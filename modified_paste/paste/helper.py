from typing import List
from anndata import AnnData
import numpy as np
import scipy
from sklearn.neighbors import KDTree

import time


import ot

from tqdm import tqdm
import torch



def kl_divergence_corresponding_backend(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    nx = ot.backend.get_backend(X,Y)

    X = X/nx.sum(X,axis=1, keepdims=True)
    Y = Y/nx.sum(Y,axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum('ij,ij->i',X,log_X)
    X_log_X = nx.reshape(X_log_X,(1,X_log_X.shape[0]))

    X_log_Y = nx.einsum('ij,ij->i',X,log_Y)
    X_log_Y = nx.reshape(X_log_Y,(1,X_log_Y.shape[0]))
    D = X_log_X.T - X_log_Y.T
    return nx.to_numpy(D)

def jensenshannon_distance_1_vs_many_backend(X, Y):
    """
    Returns pairwise Jensenshannon distance (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    assert X.shape[0] == 1
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nx = ot.backend.get_backend(X,Y)        # np or torch depending upon gpu availability
    X = nx.concatenate([X] * Y.shape[0], axis=0) # broadcast X
    X = X/nx.sum(X,axis=1, keepdims=True)   # normalize
    Y = Y/nx.sum(Y,axis=1, keepdims=True)   # normalize
    M = (X + Y) / 2.0
    kl_X_M = torch.from_numpy(kl_divergence_corresponding_backend(X, M))
    kl_Y_M = torch.from_numpy(kl_divergence_corresponding_backend(Y, M))
    js_dist = nx.sqrt((kl_X_M + kl_Y_M) / 2.0).T[0]
    return js_dist

def jensenshannon_divergence_backend(X, Y):
    """
    This function is added ny Nuwaisir
    
    Returns pairwise JS divergence (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    print("Calculating cost matrix")

    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    nx = ot.backend.get_backend(X,Y)
    # nx = ot.backend.NumpyBackend()

    # X = X.cpu().detach().numpy()
    # Y = Y.cpu().detach().numpy()

    # print(nx.unique(nx.isnan(X)))
    # print(nx.unique(nx.isnan(Y)))
        
    
    X = X/nx.sum(X,axis=1, keepdims=True)
    Y = Y/nx.sum(Y,axis=1, keepdims=True)

    n = X.shape[0]
    m = Y.shape[0]
    
    js_dist = nx.zeros((n, m))

    for i in tqdm(range(n)):
        js_dist[i, :] = jensenshannon_distance_1_vs_many_backend(X[i:i+1], Y)


    # if("numpy" in str(type(js_dist))):
    #     js_dist = torch.from_numpy(js_dist)
        
    print("Finished calculating cost matrix")
    # print(nx.unique(nx.isnan(js_dist)))

    if torch.cuda.is_available():
        return js_dist.numpy()
    else:
        return js_dist
    
    # print("type of js dist:" , type(js_dist))
    # return js_dist
    
    # print("vectorized jsd")
    # X = X/nx.sum(X,axis=1, keepdims=True)
    # Y = Y/nx.sum(Y,axis=1, keepdims=True)

    # mid = (X[:, None] + Y) / 2
    # n = X.shape[0]
    # m = Y.shape[0]
    # d = X.shape[1]
    # l = nx.ones((n, m, d)) * X[:, None, :]
    # r = nx.ones((n, m, d)) * Y[None, :, :]
    # l_2d = nx.reshape(l, (-1, l.shape[2]))
    # r_2d = nx.reshape(r, (-1, r.shape[2]))
    # m_2d = (l_2d + r_2d) / 2.0
    # kl_l_m = kl_divergence_corresponding_backend(l_2d, m_2d)
    # kl_r_m = kl_divergence_corresponding_backend(r_2d, m_2d)

    # js_dist = nx.sqrt((kl_l_m + kl_r_m) / 2.0)
    # return nx.reshape(js_dist, (n, m))


def filter_for_common_genes(
    slices: List[AnnData]) -> None:
    """
    Filters for the intersection of genes between all slices.

    Args:
        slices: List of slices.
    """
    assert len(slices) > 0, "Cannot have empty list."

    common_genes = slices[0].var.index
    for s in slices:
        common_genes = intersect(common_genes, s.var.index)
    for i in range(len(slices)):
        slices[i] = slices[i][:, common_genes]
    print('Filtered all slices for common genes. There are ' + str(len(common_genes)) + ' common genes.')

def match_spots_using_spatial_heuristic(
    X,
    Y,
    use_ot: bool = True) -> np.ndarray:
    """
    Calculates and returns a mapping of spots using a spatial heuristic.

    Args:
        X (array-like, optional): Coordinates for spots X.
        Y (array-like, optional): Coordinates for spots Y.
        use_ot: If ``True``, use optimal transport ``ot.emd()`` to calculate mapping. Otherwise, use Scipy's ``min_weight_full_bipartite_matching()`` algorithm.

    Returns:
        Mapping of spots using a spatial heuristic.
    """
    n1,n2=len(X),len(Y)
    X,Y = norm_and_center_coordinates(X),norm_and_center_coordinates(Y)
    dist = scipy.spatial.distance_matrix(X,Y)
    if use_ot:
        pi = ot.emd(np.ones(n1)/n1, np.ones(n2)/n2, dist)
    else:
        row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(scipy.sparse.csr_matrix(dist))
        pi = np.zeros((n1,n2))
        pi[row_ind, col_ind] = 1/max(n1,n2)
        if n1<n2: pi[:, [(j not in col_ind) for j in range(n2)]] = 1/(n1*n2)
        elif n2<n1: pi[[(i not in row_ind) for i in range(n1)], :] = 1/(n1*n2)
    return pi

def kl_divergence(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    X = X/X.sum(axis=1, keepdims=True)
    Y = Y/Y.sum(axis=1, keepdims=True)
    log_X = np.log(X)
    log_Y = np.log(Y)
    X_log_X = np.matrix([np.dot(X[i],log_X[i].T) for i in range(X.shape[0])])
    D = X_log_X.T - np.dot(X,log_Y.T)
    return np.asarray(D)

def kl_divergence_backend(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    nx = ot.backend.get_backend(X,Y)

    X = X/nx.sum(X,axis=1, keepdims=True)
    Y = Y/nx.sum(Y,axis=1, keepdims=True)

    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum('ij,ij->i',X,log_X)
    X_log_X = nx.reshape(X_log_X,(1,X_log_X.shape[0]))
    D = X_log_X.T - nx.dot(X,log_Y.T)
    return nx.to_numpy(D)


def intersect(lst1, lst2):
    """
    Gets and returns intersection of two lists.

    Args:
        lst1: List
        lst2: List

    Returns:
        lst3: List of common elements.
    """

    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

def norm_and_center_coordinates(X):
    """
    Normalizes and centers coordinates at the origin.

    Args:
        X: Numpy array

    Returns:
        X_new: Updated coordiantes.
    """
    return (X-X.mean(axis=0))/min(scipy.spatial.distance.pdist(X))

def apply_trsf(
    M: np.ndarray,
    translation: List[float],
    points: np.ndarray) -> np.ndarray:
    """
    Apply a rotation from a 2x2 rotation matrix `M` together with
    a translation from a translation vector of length 2 `translation` to a list of
    `points`.

    Args:
        M (nd.array): A 2x2 rotation matrix.
        translation (nd.array): A translation vector of length 2.
        points (nd.array): A nx2 array of `n` points 2D positions.

    Returns:
        (nd.array) A nx2 matrix of the `n` points transformed.
    """
    if not isinstance(translation, np.ndarray):
        translation = np.array(translation)
    trsf = np.identity(3)
    trsf[:-1, :-1] = M
    tr = np.identity(3)
    tr[:-1, -1] = -translation
    trsf = trsf @ tr

    flo = points.T
    flo_pad = np.pad(flo, ((0, 1), (0, 0)), constant_values=1)
    return ((trsf @ flo_pad)[:-1]).T



def get_niche_distribution_old(curr_slice, radius):
    """
    This method is added by Anup Bhowmik
    Args:
        curr_slice: Slice to get niche distribution for.
        pairwise_distances: Pairwise distances between cells of a slice.
        radius: Radius of the niche.

    Returns:
        niche_distribution: Niche distribution for the slice.
    """
    print("old ")
    print ("radius", radius)
    xy = curr_slice.obsm['spatial']
    tree = KDTree(xy, leaf_size=2)

    time_kd_start = time.time()
    tree = KDTree(xy, leaf_size=2, metric='euclidean')
    dist, ind = tree.query(xy, k=50)
    time_kd_end = time.time()
    # print("time taken for kd tree", time_kd_end-time_kd_start)

    time_cell_type_start = time.time()
    unique_cell_types = np.array(list(curr_slice.obs['cell_type_annot'].unique()))
    cell_type_to_index = dict(zip(unique_cell_types, list(range(len(unique_cell_types)))))
    cells_within_radius = np.zeros((curr_slice.shape[0], len(unique_cell_types)), dtype=float)
    time_cell_type_end = time.time()
    # print("time taken for cell type", time_cell_type_end-time_cell_type_start)


    for i in tqdm(range(curr_slice.shape[0])):
        # find the indices of the cells within the radius
        
        for j in range(len(ind[i])):
            
            if dist[i][j] <= radius:
               
                start1 = time.time()
                cell_type_str_j = str(curr_slice.obs['cell_type_annot'].values[ind[i][j]])
                cell_type_str_i = str(curr_slice.obs['cell_type_annot'].values[i])
                end1 = time.time()


                cells_within_radius[i][cell_type_to_index[cell_type_str_j]] += 1
                cells_within_radius[j][cell_type_to_index[cell_type_str_i]] += 1
                end = time.time()

                # print("time taken for cell type", end1-start1)
                # print("time taken for adding", end-end1)
                # print("=========================================")


    return np.array(cells_within_radius)


from sklearn.metrics.pairwise import euclidean_distances
def get_niche_distribution(curr_slice, radius):
    """
    This method is added by Anup Bhowmik
    Args:
        curr_slice: Slice to get niche distribution for.
        pairwise_distances: Pairwise distances between cells of a slice.
        radius: Radius of the niche.

    Returns:
        niche_distribution: Niche distribution for the slice.
    """

    print ("radius", radius)

    unique_cell_types = np.array(list(curr_slice.obs['cell_type_annot'].unique()))
    cell_type_to_index = dict(zip(unique_cell_types, list(range(len(unique_cell_types)))))
    cells_within_radius = np.zeros((curr_slice.shape[0], len(unique_cell_types)), dtype=float)

    # print("time taken for cell type", time_cell_type_end-time_cell_type_start)

    source_coords = curr_slice.obsm['spatial']
    distances = euclidean_distances(source_coords, source_coords)

    for i in tqdm(range(curr_slice.shape[0])):
        # find the indices of the cells within the radius

        target_indices = np.where(distances[i] <= radius)[0]
        # print("i", i)
        # print(target_indices)

        for ind in target_indices:
            cell_type_str_j = str(curr_slice.obs['cell_type_annot'][ind])
            cells_within_radius[i][cell_type_to_index[cell_type_str_j]] += 1

    return np.array(cells_within_radius)


## Covert a sparse matrix into a dense np array
to_dense_array = lambda X: X.toarray() if isinstance(X,scipy.sparse.csr.spmatrix) else np.array(X)

## Returns the data matrix or representation
extract_data_matrix = lambda adata,rep: adata.X if rep is None else adata.obsm[rep]