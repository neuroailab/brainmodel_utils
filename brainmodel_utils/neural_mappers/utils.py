import numpy as np


def map_from_str(map_type):
    if map_type.lower() == "pls":
        from brainmodel_utils.neural_mappers import PLSNeuralMap

        return PLSNeuralMap
    elif map_type.lower() == "percentile":
        from brainmodel_utils.neural_mappers import PercentileNeuralMap

        return PercentileNeuralMap
    elif map_type.lower() == "factored":
        from brainmodel_utils.neural_mappers import FactoredNeuralMap

        return FactoredNeuralMap
    elif map_type.lower() == "identity":
        from brainmodel_utils.neural_mappers import IdentityNeuralMap

        return IdentityNeuralMap
    elif map_type.lower() == "sklinear":
        from brainmodel_utils.neural_mappers import SKLinearNeuralMap

        return SKLinearNeuralMap
    else:
        raise ValueError(f"{map_type.lower()} is not supported.")


def generate_train_test_splits(
    num_stim, num_splits=5, train_frac=0.8, start_seed=0,
):
    if train_frac > 0:
        train_test_splits = []
        for s in range(start_seed, start_seed + num_splits):
            rand_idx = np.random.RandomState(seed=s).permutation(num_stim)
            num_train = (int)(np.ceil(train_frac * len(rand_idx)))
            train_idx = rand_idx[:num_train]
            test_idx = rand_idx[num_train:]

            curr_sp = {"train": train_idx, "test": test_idx}
            train_test_splits.append(curr_sp)
    else:
        print("Train fraction is 0, make sure your map has no parameters!")
        # we apply no random permutation in this case as there is no training of parameters
        # (e.g. rsa)
        train_test_splits = [
            {"train": np.array([], dtype=int), "test": np.arange(num_stim)}
        ]
    return train_test_splits
