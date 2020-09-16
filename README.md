# GPU-matching

Files:
1. ```nn.py``` - main file to run. There are 3 algorithms implemented,
 the nearest-neighbor matching from query set of descriptors to the database (DB),
 both for CPU and GPU:
    1. brute-force;
    2. Manhattan-distance based;
    3. LSH-based.

    For running on GPU ```numba``` python module is needed 
(```conda install numba```)

2. ```plot.py``` - 
3. ```plot_descr_on_image.py``` - plots match of features from two images using saved before files 
(with ending ```.res.csv.```)
4. ```r2d2_torch``` - copy of the depository (https://github.com/naver/r2d2) for extracting R2D2 descriptors in Pytorch
(needs it to be installed, or already created files with descriptors could be used )
5. ```r2d2_torch/imgs/msk[2].png.*.r2d2``` - descriptors extracted by R2D2, used for test run

To run, simply:

```python nn.py -t TYPE -m MODE --rep N```

where 

```TYPE``` is from ("cpu", "gpu")

```MODE``` is from ("brute", "manh", "hash")

```N``` - number to repeat (for collecting statistics, default 1).

Other parameters include: 

1. using all images mentioned in code or only one for DB construction - ```--add_dbs 1```.
2. if needed to save coordinates of correspondences ```--coords_save 1```.

For extracting ```K``` features run from ```r2d2_torch```:
 
```python extract.py --model models/r2d2_WASF_N8_big.pt --images imgs/ms.png --top-k K```

Technical details:

1. If needed only one set of parameters (the best), with the additional argument ```--final 1``` 
will be taken only one specified in the code set.
2. The results are saved as .csv files in subfolder ```results``` with names ```{type}_{mode}.csv``` 
and ```{type}_{mode}.final.csv``` if it was the final one (with corresp. argument). 
For further details see argument of running ```nn.py```.
3. Maximal allowed number of descriptors in DB is hard-coded in functions 
```find_dist_linked``` and ```find_dist_manh```.