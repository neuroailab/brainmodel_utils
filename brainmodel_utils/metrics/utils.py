import numpy as np
import xarray as xr
from scipy.stats import pearsonr, spearmanr
from functools import partial
from collections import defaultdict


def upper_tri(X):
    """Returns upper triangular part due to symmetry of RSA.
        Excludes diagonal as generally recommended:
        https://www.sciencedirect.com/science/article/pii/S1053811916308059"""
    return X[np.triu_indices_from(X, k=1)]


def rsm(X):
    return np.corrcoef(X)


def rdm(X):
    return 1 - rsm(X)


def input_checker_2d(X, Y):
    assert X.ndim == 2
    if isinstance(X, xr.DataArray):
        # provide an extra layer of security for xarrays
        assert X.dims[0] == "frame_id"
        assert X.dims[1] == "units"
    assert np.isfinite(X).all()

    assert Y.ndim == 2
    if isinstance(Y, xr.DataArray):
        # provide an extra layer of security for xarrays
        assert Y.dims[0] == "frame_id"
        assert Y.dims[1] == "units"
    assert np.isfinite(Y).all()


def sphalf_input_checker(X, Y, X1, X2, Y1, Y2, dim_val=2):
    if dim_val == 2:
        input_checker_2d(X=X, Y=Y)

    assert X.ndim == dim_val
    assert Y.ndim == dim_val
    assert X1.ndim == dim_val
    assert X2.ndim == dim_val
    assert Y1.ndim == dim_val
    assert Y2.ndim == dim_val
    # ensure number of stimuli is the same
    assert X.shape[0] == Y.shape[0]
    # split halves
    assert X.shape == X1.shape
    assert X1.shape == X2.shape
    assert Y.shape == Y1.shape
    assert Y1.shape == Y2.shape
    # assert finiteness
    assert np.isfinite(X1).all()
    assert np.isfinite(X2).all()
    assert np.isfinite(Y1).all()
    assert np.isfinite(Y2).all()


def rsa(X, Y, mat_type="rdm", metric="pearsonr"):
    input_checker_2d(X=X, Y=Y)

    if mat_type.lower() == "rdm":
        mat_func = rdm
    elif mat_type.lower() == "rsm":
        mat_func = rsm
    else:
        raise ValueError
    rep_X = upper_tri(mat_func(X))
    rep_X = rep_X.flatten()
    rep_Y = upper_tri(mat_func(Y))
    rep_Y = rep_Y.flatten()
    metric_func = str_to_metric_func(metric)
    return metric_func(rep_X, rep_Y)


def str_to_metric_func(name):
    if name == "pearsonr":
        metric_func = pearsonr
    elif name == "spearmanr":
        metric_func = spearmanr
    elif name == "rsa_pearsonr":
        metric_func = partial(rsa, metric="pearsonr")
    elif name == "rsa_spearmanr":
        metric_func = partial(rsa, metric="spearmanr")
        raise ValueError
    return metric_func


def concat_dict_sp(
    results_arr,
    partition_names=["train", "test"],
    agg_func=None,
    agg_func_axis=0,
    xarray_target=None,
    xarray_dims=None,
):

    results_dict = {}
    for p in partition_names:
        results_dict[p] = defaultdict(list)

    for res in results_arr:  # e.g. of length num_train_test_splits
        if res is not None:
            for p in partition_names:
                for metric_name, metric_value in res[p].items():
                    assert not isinstance(metric_value, dict)
                    results_dict[p][metric_name].append(metric_value)

    for p, v1 in results_dict.items():
        for metric_name, metric_value in v1.items():
            # e.g. turn list into np array of train_test_splits x neurons
            metric_value_concat = np.stack(metric_value, axis=0)
            if xarray_target is not None:
                assert xarray_dims is not None
                assert isinstance(xarray_dims, list)
                # the main thing that is in common are the units in the last dimension for regression,
                # so we add that metadata from the original target xarray
                assert xarray_dims[-1] == "units"
                assert xarray_target.shape[-1] == len(xarray_target.units)
                assert metric_value_concat.shape[-1] == xarray_target.shape[-1]
                metric_value_concat = xr.DataArray(
                    metric_value_concat,
                    dims=xarray_dims,
                    coords=xarray_target.units.coords,
                )

            if agg_func is not None:
                metric_value_concat = agg_func(metric_value_concat, axis=agg_func_axis)
            results_dict[p][metric_name] = metric_value_concat

    return results_dict


def generic_trial_avg(source, trial_dim="trials", trial_axis=0):
    assert source.ndim == 3
    if isinstance(source, xr.DataArray):
        X = source.mean(dim=trial_dim, skipna=True)
    else:
        X = np.nanmean(source, axis=trial_axis)
    return X


def get_splithalves(M, seed):
    if M.ndim == 2:
        # model features are deterministic
        return M, M
    else:
        assert M.ndim == 3
        rng = np.random.RandomState(seed=seed)
        n_rep = M.shape[0]
        ri = list(range(n_rep))
        rng.shuffle(ri)  # without replacement
        sphf_n_rep = n_rep // 2
        ri1 = ri[:sphf_n_rep]
        ri2 = ri[sphf_n_rep:]
        return generic_trial_avg(M[ri1]), generic_trial_avg(M[ri2])
