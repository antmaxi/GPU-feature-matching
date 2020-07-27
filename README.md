# GPU-matching

Files:
1. ```process.py``` - CPU brute--force matching

2. ```process_cuda.py``` - GPU brute--force matching, needs ''numba'' to be installed

3. ```r2d2_torch/imgsbrooklyn.png.r2d2``` - 5000 descriptors extracted by R2D2 (https://github.com/naver/r2d2), used for test run

To run, simply:

```python process[_cuda].py``` 
