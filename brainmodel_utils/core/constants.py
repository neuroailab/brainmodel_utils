import numpy as np

RIDGECV_ALPHA_CV = np.sort(
    np.concatenate(
        [
            np.geomspace(1e-9, 1e5, num=15, endpoint=True),
            5 * np.geomspace(1e-4, 1e4, num=9, endpoint=True),
        ]
    )
)
