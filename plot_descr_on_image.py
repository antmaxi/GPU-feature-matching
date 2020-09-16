import os
import matplotlib.pyplot as plt
import numpy as np
import csv
from skimage.feature import plot_matches

import argparse

parser = argparse.ArgumentParser(description='Choose the type of execution.')
parser.add_argument('-t', #required=True,
                    # help='type, "cpu" or "gpu"',
                    choices=["cpu", "gpu"])
parser.add_argument('-m', #required=True,
                    help='algorithm, "brute" or "hash" or "manh"',
                    choices=["brute", "manh", "hash"])
parser.add_argument('--save',  type=int,
                    help='if need to save picture (otherwise just show)',
                    default=0)

args = parser.parse_args()
if 1:
    args.t = "gpu"
    args.m = "manh"
    args.save = 1
x_query = []
y_query = []
x_db = []
y_db = []

name_query = "mipt_4.png"
name_db = "mipt_5.png"
query = "r2d2_torch/imgs/" + name_query
db = "r2d2_torch/imgs/" + name_db

filename = 'results/' + args.t + '_' + args.m + "_res" + '.csv'
with open(str(filename), 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        x_query.append(float(row[0]))
        y_query.append(float(row[1]))
        x_db.append(float(row[2]))
        y_db.append(float(row[3]))
im_query = plt.imread(query)
im_db = plt.imread(db)
fig, ax = plt.subplots(nrows=1, ncols=1)#, figsize=(12,4))
N = len(x_query)
matches = np.zeros((N, 2), dtype=int)
for i in range(N):
    matches[i, :] = [i, i]
N_to_show = 30
plot_matches(ax, im_query, im_db,
             np.column_stack((y_query[0:N_to_show], x_query[0:N_to_show])),
             np.column_stack((y_db[0:N_to_show], x_db[0:N_to_show])),
             matches[0:N_to_show, :])
ax.axis("off")
ax.set_title("{}_{}, {} closest correspondences".format(args.t, args.m, N_to_show))
if args.save:
    plt.savefig("/home/anton/Documents/ETH-Zurich-projects/Seminar in Robotics/corresp_"
                + args.t + "_" + args.m + "_4-5.png",
                bbox_inches='tight',
                dpi=300)
else:
    plt.show()
#plt.imshow(im_query)
#plt.scatter(x_query, y_query,  s=3, c="r")
