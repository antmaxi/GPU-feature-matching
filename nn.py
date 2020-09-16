import sys
import os
import numpy as np
from numpy import matlib
import time
import math
import csv
import argparse

import numba
from numba import cuda
from numba import jit

@cuda.jit
def find_dist_brute(descr, db, d_N, d_N_db, d_F, closest):
    i = cuda.grid(1)
    closest_ind = None
    N = d_N[0]
    F = d_F[0]
    N_db = d_N_db[0]
    dist = 1e20
    if i < N:
        for j in range(N_db):
            d = 0
            for k in range(F):
                d_i = descr[i][k] - db[j][k]
                d = d + d_i * d_i
            if d < dist:
                dist = d
                closest_ind = j
        closest[i] = closest_ind


@cuda.jit
def find_dist_hash(descr, db, h_query, h_db, d_a, d_b, d_M, d_W, d_N, d_N_db, d_F, closest):
    # find closest candidates using Manhattan distance of binned distances to planes
    i = cuda.grid(1)
    MAX_DIST = 1e20
    EPS = 1e-8
    dist = MAX_DIST
    closest_ind = None
    N_db = d_N_db[0]
    F = d_F[0]
    M = d_M[0]
    N = d_N[0]
    #result = np.zeros(10)
    # TODO: calculate h_current again? (as for new feature) - no, on CPU very fast
    W = d_W[0]
    #h_curr = np.floor((a @ descr[i] + b) / W)
    best_ind = numba.cuda.local.array(30000, numba.boolean) # TODO: N doesn't work, why?
    for k in range(N_db):
        best_ind[k] = False
    # print(numba.typeof(N))
    if i < N:
        # find closest candidates using Manhattan distance of binned distances to planes
        for j in range(N_db):
            d = 0
            for k in range(M):
                d_i = h_query[i][k] - h_db[j][k]
                d = d + abs(d_i)
            if d < dist:
                dist = d
                for k in range(N_db):
                    best_ind[k] = False
                best_ind[j] = True
            elif abs(d - dist) < EPS:
                best_ind[j] = True
        # check Euclidean distances to candidates
        dist = MAX_DIST
        for j in range(len(best_ind)):
            if best_ind[j]:
                d = 0
                for k in range(F):
                    d_i = descr[i][k] - db[j][k]
                    d = d + d_i * d_i
                if d < dist:
                    dist = d
                    closest_ind = j
        closest[i] = closest_ind


@cuda.jit
def find_dist_linked(descr, db, hash_query, linked,
                    #d_M, d_W,
                    d_N, d_N_db,
                    d_F, d_L, d_tableSize,
                    closest,):
    # find candidates using precomputed linked list of appearances of DB descriptors using their hashed distances
    # to the set of random planes
    i = cuda.grid(1)
    MAX_DIST = 1e20
    closest_ind = None
    N = d_N[0]
    N_db = d_N_db[0]
    F = d_F[0]
    L = d_L[0]
    tableSize = d_tableSize[0]
    best_ind = numba.cuda.local.array(30000, numba.boolean)  # TODO: N doesn't work, why?
    for k in range(N_db):
        best_ind[k] = False
    if i < N:
        # find candidates using precomputed linked list
        for l in range(L):
            curr = 2 * (l * tableSize + hash_query[i, l])
            next_el = linked[curr + 1]
            while next_el:
                best_ind[linked[next_el]] = True
                next_el = linked[next_el + 1]
        # check Euclidean distances to candidates
        dist = MAX_DIST
        for j in range(len(best_ind)):
            if best_ind[j]:
                d = 0
                for k in range(F):
                    d_i = descr[i][k] - db[j][k]
                    d = d + d_i * d_i
                if d < dist:
                    dist = d
                    closest_ind = j
        closest[i] = closest_ind


def r2d2_call(filename, pic_name, N_needed):
    cwd = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isfile(filename):
        print("File {} doesn't exist, running R2D2".format(filename))
        os.system("python " + cwd + "/r2d2_torch/extract.py --model " + cwd + "/r2d2_torch/models/r2d2_WASF_N8_big.pt "
                  + "--images " + cwd + "/r2d2_torch/imgs/" + pic_name + ".png --top-k " + str(N_needed)
                  + " --min-size 0 --max-size 9999 --min-scale 0.3 --max-scale 1.0")
    else:
        pass  # print("All files exist")


# OLD VERSION, NOT USED
@cuda.jit
def find_dist_hash2(descr, db, h_query, h_db,
                    #d_a, d_b,  d_M, d_W, d_N_db,
                    d_F, d_L, #d_tableSize,
                    closest,):
    # OLD VERSION
    # find closest candidates using matching in hashed distances to random planes
    i = cuda.grid(1)
    MAX_DIST = 1e20
    closest_ind = None
    N_db = d_N_db[0]
    F = d_F[0]
    L = d_L[0]
    best_ind = numba.cuda.local.array(30000, numba.boolean)  # TODO: N doesn't work, why?
    for k in range(N_db):
        best_ind[k] = False
    if i < N:
        # find closest candidates using hashing, first from matching bins, then calculating Euclidean distance
        for j in range(N_db):
            d = 0
            for k in range(L):
                if h_query[i][k] == h_db[j][k]:
                    best_ind[j] = True
        # check Euclidean distances to candidates
        dist = MAX_DIST
        for j in range(len(best_ind)):
            if best_ind[j]:
                d = 0
                for k in range(F):
                    d_i = descr[i][k] - db[j][k]
                    d = d + d_i * d_i
                if d < dist:
                    dist = d
                    closest_ind = j
        closest[i] = closest_ind
########################################################################################################################
# maximuim requested to R2D2 number of descriptors in DB
NUM_DESCR_IN_DB = int(5e3)
# default number of test descriptors
NUM_DESCR_IN_QUERY = int(5e3)

##  FILES TO USE AS QUERY AND DATABASE (if only one) ##########
query_name = "mipt_4"
db_name = "mipt_5"

parser = argparse.ArgumentParser(description='Choose the type of execution.')
parser.add_argument('-t', required=True, help='type, "cpu" or "gpu"',
                    choices=["cpu", "gpu"])
parser.add_argument('-m', required=True, help='algorithm, "brute" or "hash" or "manh"',
                    choices=["brute", "manh", "hash"])

parser.add_argument('--query_name', help='name of the picture to use descriptors',
                    default=query_name)
parser.add_argument('--db_name', help='name of the picture to extract database',
                    default=db_name)

parser.add_argument('--db', help='path to the database file',
                    default="r2d2_torch/imgs/" + db_name + ".png." + str(int(NUM_DESCR_IN_DB))
                            + ".r2d2")
parser.add_argument('--query', help='path to the query descriptors file',
                    default="r2d2_torch/imgs/" + query_name + ".png." + str(int(NUM_DESCR_IN_QUERY)) + ".r2d2")

parser.add_argument('--rep', type=int, help='numbers of repetitions (for statistics)',
                    default=1)
parser.add_argument('--final', type=int, help='If needed to use only one set of parameters (the best)',
                    default=0)

parser.add_argument('--add_dbs', type=int, help='If add descriptors from several images to construct big DB',
                    default=0)
parser.add_argument('--db_folder',  help='folder where images are stored',
                    default="/r2d2_torch/imgs/")
parser.add_argument('--coord_save',  type=int,
                    help='if need to save coordinates of descriptors and their correspondence',
                    default=0)

args = parser.parse_args()
db_folder = args.db_folder
repetitions = args.rep

########################################################################################################################
test_numbers = (
                5000,
                #300, 500,
                #1000,
                #3000,
                #5000,
                #8000,
                #10000, 30000,
                #50000,
                #80000,
                #100000,
                )
N_db_numbers = (
                #1000,
                5000,
               #10000, 20000, 30000,
               #50000, 100000, 150000, 200000, 300000,
               )
if args.final:
    if args.add_dbs:
        test_numbers = (5000,)
    else:
        test_numbers = (100,
                        500,
                        1000, 3000,
                        5000,
                        7000,
                        10000, 15000,
                        )
    test_numbers = (5000,)
if args.add_dbs:
    db_list = ["mipt_" + str(i + 1) for i in range(16)]
    print(db_list)
else:
    db_list = (args.db_name,)
for k, db_name_current in enumerate(db_list):
    database_name = db_folder + db_name_current + ".png" \
                    + "." + str(NUM_DESCR_IN_DB) + ".r2d2"
    # Load database
    print("Database file: " + database_name)

    r2d2_call(os.getcwd() + database_name, db_name_current, NUM_DESCR_IN_DB)
    with np.load(os.getcwd() + database_name) as data:
        if k == 0:
            db_all = data["descriptors"]
        else:
            db_all = np.vstack((db_all, data["descriptors"]))
        coords_db = data["keypoints"]

N_db_all, F = np.shape(db_all)  # number of database descriptors, their size
print("Real volume of the database: {}".format(N_db_all))
#####################################################################3
#for args.t in ("gpu",):
for x1 in range(1):  # auxiliary loops just to replace args.t and args.m ones, when modes are set automatically
    #for args.m in ("brute",# "manh", "hash"
    #              ):
    for x2 in range(1):
        # test with different numbers of descriptors in query
        times_final_mean = []
        times_final_std = []
        dist_mean = []
        dist_std = []
        test_numbers_brute = []
        db_numbers = []
        # variables for hash parameters selection:
        ms = []
        ws = []
        ls = []
        tableSizes = []
        test_numbers_hash = []
        # Main loop
        for N_descr in test_numbers:
            for N_db in N_db_numbers:
                if N_db > N_db_all:
                    break
                print(args.m, args.t)
                print(N_db, N_db_all)
                # take only needed number in DB
                db = db_all[0:N_db]
                #if x2 >= 4:
                #    db = db_all[N_db:3*N_db]
                test_name = db_folder + args.query_name + ".png." + str(int(N_descr)) + ".r2d2"
                # code to run r2d2 automatically, if don't have appropriate descriptors number yet as a separate file
                # TODO: maybe just take subset from big file instead of creating many files?
                r2d2_call(os.getcwd() + test_name, args.query_name, N_descr)
                print("Test file:     " + test_name)
                with np.load(os.getcwd() + test_name) as data:
                    descr = data["descriptors"]
                    coords_query = data["keypoints"]
                N = np.shape(descr)[0]  # number of query descriptors
                print("Real number of the query descriptors: {}".format(N))

                if args.m == "brute":
                    times_temp = []
                    if args.t == "cpu":
                        for rep in range(repetitions):
                            start = time.time()
                            res = []
                            for i in range(N):
                                dist = np.linalg.norm(db - matlib.repmat(descr[i,:], np.shape(db)[0], 1),
                                                      axis=1)
                                res.append(int(np.argmin(dist)))
                            end = time.time()
                            times_temp.append(end - start)
                    elif args.t == "gpu":
                        threads_per_block = 128
                        blocks_per_grid = math.ceil(N / threads_per_block)
                        d_db = cuda.to_device(db)
                        d_N_db = cuda.to_device(N_db)
                        d_F = cuda.to_device(F)
                        for rep in range(repetitions):
                            # start search
                            start = time.time()
                            d_descr = cuda.to_device(descr)
                            # TODO: could we do it without sending dimensions? (without numpy on devices?) And without sending all test descriptors?
                            d_N = cuda.to_device(N)
                            glob_closest = cuda.device_array(shape=(N,), dtype=np.int32)
                            find_dist_brute[blocks_per_grid, threads_per_block](d_descr, d_db, d_N, d_N_db, d_F, glob_closest)
                            res = glob_closest.copy_to_host()
                            end = time.time()
                            times_temp.append(end - start)
                            print(end-start)
                    # Write results #############################
                    t_temp = np.array(times_temp)
                    print(times_temp)
                    if args.t == "gpu":
                        times_final_mean.append(float('%.3f' % (np.mean(t_temp[1:]))))
                        times_final_std.append(float('%.3f' % (np.std(t_temp[1:]))))
                        print('t={:.2f}, std={:.3f}'.format(np.mean(t_temp[1:]), np.std(t_temp[1:])))
                    else:
                        times_final_mean.append(float('%.3f' % (np.mean(t_temp))))
                        times_final_std.append(float('%.3f' % (np.std(t_temp))))
                        print('t={:.2f}, std={:.3f}'.format(np.mean(t_temp), np.std(t_temp[1:])))
                    # TODO: create list of distances to the best, then compare with hashing => find some super-bad corresp. maybe?
                    distances_final = []
                    for i in range(len(res)):
                        distances_final.append(np.linalg.norm(descr[i, :]
                                                              - db[res[i], :]))
                    dist_mean.append(float('%.3f' % (np.mean(np.array(distances_final)))))
                    test_numbers_brute.append(N)
                    db_numbers.append(N_db)
                    # write correspondences to the file for the last repetition
                    if args.coord_save:
                        x1 = coords_query[:, 0].tolist()
                        y1 = coords_query[:, 1].tolist()
                        x2 = coords_db[res, 0].tolist()
                        y2 = coords_db[res, 1].tolist()
                        filename_coords = 'results/' + args.t + '_' + args.m + "_res" + '.csv'
                        with open(filename_coords, 'w') as f:
                            writer = csv.writer(f)
                            writer.writerows(sorted(zip(x1, y1, x2, y2, distances_final),
                                                    key=lambda t: t[4]))
                            print("Saved corresponding coordinates in {}".format(filename_coords))

                    if 0:
                        with open('results_' + args.m + '_' + args.t + '.txt', 'w') as f:
                            f.write('mean={:.2f}, std={:.2f}\n'.format(np.mean(t), np.std(t)))
                            for i in range(len(t)):
                                print('t={:.2f}'.format(t[i]))
                                f.write('t={:.2f}\n'.format(t[i]))
                            times_last.append(float('%.3f' % (t[len(t) - 1])))

                elif args.m == "manh" or args.m == "hash":
                    # M - number of planes to use
                    # more -> less selective
                    M_list = (
                              1,
                              #2,
                              #3,
                              #7,
                              #15,
                              #330,
                              #50,
                              )
                    # W - size of bin (cut) when coding (cutting) distance from plane to the descriptor
                    # more -> less selective
                    W_list = (
                              #0.003,
                              0.01,
                              #0.1,
                              #0.5,
                              #1,
                              #3,
                              #5,
                              #10,
                              )
                    # L - number of constructed M-long codings (L=1 in "hash" mode)
                    # more -> less selective
                    L_list = (
                              #1,
                              #33,
                              #67,
                              #10,
                              3,
                              #             1,
                              )
                    # size of the hashing table when using hashes (in "hash2" mode)
                    # more -> more selective
                    tableSize_list = (
                                        #3,
                                        #7,
                                        17,
                                        37,
                                        #83,
                                     )
                    # if needed only one, "the best" set for further comparison of modes
                    if args.final:
                        if args.m == "manh":
                            M_list = (1
                                      ,)
                            W_list = (1
                                      ,)
                        elif args.m == "hash":
                            L_list = (3
                                      ,)
                            tableSize_list = (83
                                              ,)
                            M_list = (1
                                      ,)
                            W_list = (0.1
                                      ,)

                    for M in M_list:
                        for W in W_list:
                            if args.m == "manh":
                                L_list = (0,)
                                tableSize_list = (0,)
                            for L in L_list:
                                for tableSize in tableSize_list:
                                    print(M, W, L, tableSize)
                                    times_hash = []
                                    distances_final = []
                                    for rep in range(repetitions):
                                        # with hashing of binary plane-codes
                                        if args.m == "hash":
                                                # get random plane and offset
                                                a = np.random.normal(size=(L, M, F))
                                                b = np.random.uniform(0, W, (L, M, 1))
                                                # hashing for database
                                                h_db = np.floor((np.dot(a, np.transpose(db)) + b) / W).astype(int)  # L x M x N_db
                                                h_db = np.transpose(h_db, axes=[2, 0, 1])  # N_db x L x M
                                                coeff_hash = np.random.randint(low=10, size=(1, M)) + np.ones(shape=(1, M))
                                                hash_db = (np.inner(h_db, coeff_hash) % tableSize).astype(int)  # N_db x L x 1
                                                hash_db = hash_db.reshape(N_db, -1)  # N_db x L
                                                # hashing for query; is superfast, so could be done on CPU in any case
                                                h_query = np.floor((np.dot(a, np.transpose(descr)) + b) / W).astype(int)
                                                h_query = np.transpose(h_query, axes=[2, 0, 1])  # N x L x M
                                                hash_query = (np.inner(h_query, coeff_hash) % tableSize).astype(int)  # N x L x 1
                                                hash_query = hash_query.reshape(N, -1)  # N x L
                                                # constructing linked list for fast search
                                                # need L*tableSize (first access) + N_db*L
                                                linked = np.zeros(2 * (L * tableSize + N_db * L), dtype="int32")
                                                #bits = 32  # how many bits in one element (half of the place)
                                                ind_last = 2 * (L * tableSize - 1)  # until which place list is reserved
                                                for l in range(L):
                                                    for t in range(tableSize):
                                                        ind_curr = 2 * (l * tableSize + t)  # index in linked list array
                                                        for n in range(N_db):
                                                            if hash_db[n, l] == t:
                                                                ind_last += 2
                                                                linked[ind_curr] = n  # next descriptor number
                                                                linked[ind_curr + 1] = ind_last  # next position in this array
                                                                ind_curr = ind_last
                                                if args.t == "cpu":  # hash
                                                    start = time.time()
                                                    res = []
                                                    k = 0
                                                    if 0:
                                                        # find if at least in one of bins there is a correspondence with DB
                                                        for i in range(N):
                                                            ind = []
                                                            for j in range(L):
                                                                ind.extend(np.argwhere(hash_db[:, j] == hash_query[i, j]).ravel())
                                                            ind1 = np.unique(ind)
                                                            if len(ind) != 0:
                                                                dist = np.linalg.norm(db[ind1, :] - matlib.repmat(descr[i, :], len(ind1), 1),
                                                                                      axis=1)
                                                                res.append(ind1[np.argmin(dist)])
                                                                k += 1
                                                            else:
                                                                res.append(None)
                                                    else:
                                                        for i in range(N):
                                                            ind = []
                                                            for l in range(L):
                                                                curr = 2 * (l * tableSize + hash_query[i, l])
                                                                next_el = linked[curr + 1]
                                                                while next_el:
                                                                    ind.append(linked[next_el])
                                                                    next_el = linked[next_el + 1]
                                                            ind1 = np.unique(ind)
                                                            if i == 100:
                                                                print(len(ind1))
                                                            if len(ind1) != 0:
                                                                dist = np.linalg.norm(db[ind1, :] - matlib.repmat(descr[i, :], len(ind1), 1),
                                                                                      axis=1)
                                                                res.append(ind1[np.argmin(dist)])
                                                                k += 1
                                                            else:
                                                                res.append(None)
                                                    #print(res)
                                                elif args.t == "gpu":  # hash
                                                    threads_per_block = 128
                                                    blocks_per_grid = math.ceil(N / threads_per_block)
                                                    d_db = cuda.to_device(db)
                                                    d_hash_db = cuda.to_device(hash_db)
                                                    d_linked = cuda.to_device(linked)
                                                    d_N_db = cuda.to_device(N_db)
                                                    d_F = cuda.to_device(F)
                                                    #d_a = cuda.to_device(a)  # not used currently
                                                    #d_b = cuda.to_device(b)  # not used currently
                                                    #d_M = cuda.to_device(M)  # not used currently
                                                    #d_W = cuda.to_device(W)
                                                    d_L = cuda.to_device(L)
                                                    d_tableSize = cuda.to_device(tableSize)  # not used currently
                                                    times_temp = []
                                                    start = time.time()
                                                    d_hash_query = cuda.to_device(hash_query)
                                                    d_descr = cuda.to_device(descr)
                                                    d_N = cuda.to_device(N)
                                                    glob_closest = cuda.device_array(shape=(N,), dtype=np.int32)
                                                    find_dist_linked[blocks_per_grid, threads_per_block](
                                                                                              d_descr,
                                                                                              d_db,
                                                                                              d_hash_query,
                                                                                              d_linked,
                                                                                              # d_hash_db
                                                                                              # d_a, d_b,
                                                                                              # d_M, d_W,
                                                                                              d_N, d_N_db, d_F,
                                                                                              d_L, d_tableSize,
                                                                                              glob_closest,
                                                                                              )
                                                    res = glob_closest.copy_to_host()

                                                end = time.time()
                                        # with Manhattan distance check of binary plane-codes
                                        elif args.m == "manh":
                                            # get random plane and offset
                                            a = np.random.normal(size=(M, F))
                                            b = np.random.uniform(0, W, (1, M))
                                            # resulting hashes
                                            # hashing for database
                                            h_db = np.floor((db @ np.transpose(a) + b) / W)
                                            if 0:
                                                # histogram for control of adequate "M" and "W"
                                                n, bins, patches = plt.hist(h)
                                                plt.xlabel('Value')
                                                plt.ylabel('#')
                                                plt.title(r'h')
                                                plt.show()

                                            if args.t == "cpu":
                                                start = time.time()
                                                # hashing for query
                                                h_query = np.floor((descr @ np.transpose(a) + b) / W)
                                                # get closest one to every descriptor
                                                res = []
                                                for i in range(N):
                                                    # find descriptors in the same bin #and delete itself
                                                    ind = np.argwhere(np.all(h_db == h_query[i], axis=1)).ravel()
                                                    # If no other descriptors in the same bin
                                                    if len(ind) == 0:
                                                        # take closest in Manhattan distance
                                                        delta = np.sum(abs(h_db - h_query[i]), axis=1)
                                                        ind = np.argwhere(delta == np.min(delta)).ravel()
                                                    # get Euclidean distances between current set and the descriptor
                                                    dist = np.linalg.norm(db[ind] - matlib.repmat(descr[i, :], np.shape(ind)[0], 1),
                                                                          axis=1)

                                                    # print('{} {}'.format(np.argmin(dist), len(ind)))
                                                    res.append(ind[np.argmin(dist)])
                                                end = time.time()
                                            elif args.t == "gpu":  # manh
                                                threads_per_block = 128
                                                blocks_per_grid = math.ceil(N / threads_per_block)
                                                # Send everything needed to GPU
                                                d_db = cuda.to_device(db)
                                                d_h_db = cuda.to_device(h_db)
                                                d_F = cuda.to_device(F)
                                                d_a = cuda.to_device(a)
                                                d_b = cuda.to_device(b)
                                                d_M = cuda.to_device(M)
                                                d_W = cuda.to_device(W)
                                                d_N_db = cuda.to_device(N_db)

                                                start = time.time()
                                                # TODO: produce h_query on GPU for "gpu" (though it takes only ~0.01 s)
                                                # hashing for query
                                                h_query = np.floor((descr @ np.transpose(a) + b) / W)
                                                d_descr = cuda.to_device(descr)
                                                d_h_query = cuda.to_device(h_query)
                                                d_N = cuda.to_device(N)

                                                glob_closest = cuda.device_array(shape=(N,), dtype=np.int32)
                                                find_dist_hash[blocks_per_grid, threads_per_block](
                                                                                              d_descr, d_db,
                                                                                              d_h_query, d_h_db,
                                                                                              d_a, d_b,
                                                                                              d_M, d_W,
                                                                                              d_N, d_N_db, d_F,
                                                                                              glob_closest,
                                                                                              )
                                                res = glob_closest.copy_to_host()
                                                end = time.time() # remove time of sending database info to GPU
                                                #print(start1 - end1)
                                        # END OF SEARCH FOR NN #####################
                                        # Recording distances to NN
                                        for i in range(len(res)):
                                            if not res[i] is None:
                                                distances_final.append(np.linalg.norm(descr[i, :]
                                                                                      - db[res[i], :]))
                                        times_hash.append(end - start)
                                        print(times_hash)
                                    # write correspondences to the file for the last repetition
                                    if args.coord_save:
                                        x1 = coords_query[:, 0].tolist()
                                        y1 = coords_query[:, 1].tolist()
                                        x2 = coords_db[res, 0].tolist()
                                        y2 = coords_db[res, 1].tolist()
                                        filename_coords = 'results/' + args.t + '_' + args.m + "_res" + '.csv'
                                        with open(filename_coords, 'w') as f:
                                            writer = csv.writer(f)
                                            writer.writerows(sorted(zip(x1, y1, x2, y2, distances_final),
                                                                    key=lambda t: t[4]))
                                            print("Saved corresponding coordinates in {}".format(filename_coords))
                                    # Aggregated info for fixed M and W
                                    if args.t == "gpu":
                                        times_final_mean.append(float('%.3f' % (np.mean(np.array(times_hash[1:])))))
                                        times_final_std.append(float('%.3f' % (np.std(np.array(times_hash[1:])))))
                                    else:
                                        times_final_mean.append(float('%.3f' % (np.mean(np.array(times_hash)))))
                                        times_final_std.append(float('%.3f' % (np.std(np.array(times_hash)))))
                                    dist_mean.append(float('%.3f' % (np.mean(np.array(distances_final)))))
                                    dist_std.append(float('%.3f' % (np.std(np.array(distances_final)))))
                                    ls.append(L)
                                    tableSizes.append(tableSize)
                                    ms.append(M)
                                    ws.append(W)
                                    test_numbers_hash.append(N)
                                    db_numbers.append(N_db)

        if not os.path.exists('results'):
            os.makedirs('results')

        # Write final results ##############################
        final = ""
        if args.final:
            final = ".final"
        if args.add_dbs:
            final = final + ".db"
        else:
            final = final + ".query"
        with open('results/' + args.t + '_' + args.m + final + '.csv', 'w') as f:
            writer = csv.writer(f)
            print("Saved in {}".format('results/' + args.t + '_' + args.m + final + '.csv'))
            # print sorted by quality (average distance) information
            if args.m == "brute":
                writer.writerows(sorted(zip(test_numbers_brute, db_numbers, times_final_mean, dist_mean, times_final_std),
                                 key=lambda t: t[3]))
            elif args.m == "manh":

                writer.writerows(
                    sorted(zip(test_numbers_hash, db_numbers, times_final_mean, dist_mean, times_final_std, dist_std, ms, ws, ),
                           key=lambda t: t[3]))
            elif args.m == "hash":
                writer.writerows(sorted(zip(test_numbers_hash, db_numbers, times_final_mean, dist_mean, times_final_std, dist_std,
                                            ls, tableSizes, ms, ws, ),
                                        key=lambda t: t[3]))
