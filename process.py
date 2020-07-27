import numpy as np
from numpy import matlib
import time

def find_closest(descr, i):
    dist = np.linalg.norm(descr-matlib.repmat(descr[i, :], np.shape(descr)[0], 1),
                          axis=1)
    dist[i] = 1e20
    return np.argmin(dist)

pic_name = "brooklyn"
with np.load("r2d2_torch/imgs/" + pic_name + ".png.r2d2") as data:
    descr = data["descriptors"]
    start = time.time()
    print(descr)
    print(np.shape(descr))

    n_samples, n_features = np.shape(descr)

    res = []
    for i in range(n_samples):
        res.append(find_closest(descr, i))
    pairs = []
    for i in range(n_samples):
         if res[res[i]] == i and i < res[i]:
            p = [i, res[i]]
            if p not in pairs:
                pairs.append(p)
    print(pairs)

    end = time.time()
    print("Elapsed (with compilation) = %s seconds" % (end - start))
    print("Found %s correspondences (pairwise closest in Euclidean sense)" % len(pairs))
