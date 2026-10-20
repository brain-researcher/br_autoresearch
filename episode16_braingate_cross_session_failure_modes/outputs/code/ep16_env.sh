# EP16 compute environment (Sherlock modules; no installs). Source inside a Slurm allocation.
module load python/3.12.1 py-numpy/1.26.3_py312 py-scipy/1.12.0_py312
module load viz
module load py-matplotlib/3.8.3_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
