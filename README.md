# Cytnx CTMRG

## Requirements

- Working installation of Cytnx (with CUDA support)
- You need to copy the installation path of Cytnx to the `main.py` file
```python
sys.path.append('/home/petjelinux/Cytnx_lib') # This is the path to the Cytnx installation
```

## To test the performance of the code, you can use the following command:

```bash
python -m cProfile -o output.pstats main.py --chi 16 --bondim 4 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10 --GLOBALARGS_device 0 --CTMARGS_projector_svd_method GESVD
```

## To visualize the profile result, you can use the following command:

```bash
python -m gprof2dot -f pstats output.pstats | dot -T png -o profile.png
```

## To use nsys to profile the code, you can use the following command:

```bash
nsys profile --stats=true python main.py --chi 16 --bondim 4 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10 --GLOBALARGS_device 0 --CTMARGS_projector_svd_method GESVD
```

## To generate benchmark results, you can use the following command:
```python
for chi in {16,32,64,128,256} 
do
    echo "chi: ${chi}"
    python main.py --chi ${chi} --bondim 4 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10 --GLOBALARGS_device 0 --CTMARGS_projector_svd_method GESVD
done
```

And you can manually copy the results to the `benchmark.ipynb` to generate the benchmark results.