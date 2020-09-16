#plots graphs for all combinations of mode and machine
#using corresp. files from the subfolder ```results```.
#######################################################33

from scipy import optimize
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse

parser = argparse.ArgumentParser(description='Choose the type of execution.')
parser.add_argument('--logx', help='file to process, in .csv format', type=int, default=1)
parser.add_argument('--logy', help='file to process, in .csv format', type=int, default=1)
parser.add_argument('--db_or_query', help='plot dependence on db (1) or query (0) size',
                    type=int, default=1)

args = parser.parse_args()

filenames = (
            "results/cpu_brute.final",
             "results/gpu_brute.final",
             "results/cpu_manh.final",
             "results/gpu_manh.final",
             "results/cpu_hash.final",
             "results/gpu_hash.final",
)
if args.db_or_query:
    add = "db"
else:
    add = "query"
addition = "." + add + ".csv"
numbers_all = []
for filename in filenames:
    numbers = []
    times = []
    times_std = []
    with open(str(filename + addition), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if args.db_or_query:
                numbers.append(int(row[1]) / 1000)
                times.append(float(row[2]))
            else:
                numbers.append(int(row[0]) / 1000)
                times.append(float(row[1]))
            #times_std.append(float(row[4]))
        print(filename + addition)
        print(numbers, times)
        numbers_all.extend(numbers)
        name = os.path.split(filename)[1].split(".")[0]
        color = "b" if name[0:3] == "cpu" else "r"
        inv_color = "r" if name[0:3] == "cpu" else "b"
        switcher = {"brute": "v",
                    "manh": "s",
                    "hash": "o",
                    }
        form = switcher[name[4:]]
        pl_format = "-" + color + form
        if len(times_std):
            plt.plot(numbers, times, pl_format, markersize=7, markeredgecolor="k", label=name)
            #print("with errors")
            #plt.errorbar(numbers, times, yerr=times_std, fmt=pl_format, label=name, markersize=7)
        else:
            plt.plot(numbers, times, pl_format, markersize=7, markeredgecolor="k", label=name)
            #plt.errorbar(numbers, times, fmt=pl_format, label=name)
        if args.logx:
            plt.xscale('log')
        if args.logy:
            plt.yscale('log')

if args.logy:
    plt.ylabel("time, s", fontsize=14)
    #plt.ylabel(r'$\log_{10}$(time)', fontsize=14)
else:
    plt.ylabel("time, s")
if args.logx:
    if args.db_or_query:
        plt.xlabel(r'$N_{db}$', fontsize=14)
    else:
        plt.xlabel(r'$N_{query}$', fontsize=14)
    #plt.xlabel(r"$\log_{10}$|query/$10^3$|")
else:
    plt.xlabel(r"|query|, $10^3$")
numbers_all = list(set(numbers_all))
# Tweaks
if args.db_or_query:
    numbers_all.remove(150)
    numbers_all.remove(200)
print(numbers_all)
str_numbers = [str(int(numbers_all[i])) + "K" if numbers_all[i] >= 1 else str(int(numbers_all[i]*1000))
               for i in range(len(numbers_all))]
plt.xticks(numbers_all, str_numbers)
if args.db_or_query:
    pass
else:
    #plt.title("Time in dependence on the size of query")
    plt.xlim(0.08, 45)
    plt.text(17, 0.62, "hash")
    plt.text(17, 0.91, "manhattan ")
    plt.text(17, 1.8, "brute-force ")
    plt.text(17, 105, "hash")
    plt.text(17, 62, "manhattan")
    plt.text(17, 181, "brute-force")
    plt.text(1.3, 49, "CPU", rotation=0, # 33
             color="b", size=15, weight="bold")
    plt.text(1.7, 0.7, "GPU", rotation=0, color="r", size=15, weight="bold")
times_axis = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
plt.yticks(times_axis, [str(times_axis[i]) for i in range(len(times_axis))])
plt.legend()
plt.plot()
#plt.show()

if args.logx and args.logy:
    plt.savefig("results/log_yx_" + add + ".png", bbox_inches='tight')
    plt.savefig("/home/anton/Documents/ETH-Zurich-projects/Seminar in Robotics/log_yx_" + add + ".png",
                dpi=300, bbox_inches = "tight")
elif args.logy:
    plt.savefig("results/log_y" + addition + ".png", bbox_inches='tight')
else:
    plt.savefig("results/1" + addition + ".png", bbox_inches='tight')