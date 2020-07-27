import numpy as np
from numba import cuda
import time
import math

@cuda.jit
def find_dist(descr, d_N, d_F, closest):
    i = cuda.grid(1)
    dist = 1e20
    closest_ind = None
    N = d_N[0]
    F = d_F[0]
    if i < N:
        for j in range(N):
            if j != i:
                d = 0
                for k in range(F):
                    d_i = descr[i][k] - descr[j][k]
                    d = d + d_i * d_i
                if d < dist:
                    dist = d
                    closest_ind = j
        closest[i] = closest_ind

def get_closest():
    descr = data["descriptors"]
    N, F = np.shape(descr)

    threads_per_block = 128
    blocks_per_grid = math.ceil(N / threads_per_block)

    d_descr = cuda.to_device(descr)
    d_N = cuda.to_device(N)
    d_F = cuda.to_device(F)
    glob_closest = cuda.device_array(shape=(N,), dtype=np.int32)

    find_dist[blocks_per_grid, threads_per_block](d_descr, d_N, d_F, glob_closest)
    res_closest = glob_closest.copy_to_host()
    pairs = []
    for i in range(N):

        if res_closest[res_closest[i]] == i and i < res_closest[i]:
            p = [i, res_closest[i]]
            if p not in pairs:
                pairs.append(p)
    return pairs

pic_name = "brooklyn"
with np.load("r2d2_torch/imgs/" + pic_name + ".png.r2d2") as data:
    start = time.time()
    result = get_closest()
    print(result)
    end = time.time()
    print("Elapsed (with compilation) = %s seconds" % (end - start))
    print("Found %s correspondences (pairwise closest in Euclidean sense)" % len(result))
