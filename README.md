# Installation

```
module load anaconda3
module use /share/apps/spack/envs/fgci-centos7-haswell-dev/lmod/linux-centos7-x86_64/all
module load nvhpc/21.5
cd lib
make
```


# Execution

get an interactive session, larger grids (step<0.05) likely require more mem-per-cpu:

```
sinteractive -t 10 --gres=gpu:v100:1 --mem-per-cpu=2000
module load anaconda3
module use /share/apps/spack/envs/fgci-centos7-haswell-dev/lmod/linux-centos7-x86_64/all
module load nvhpc/21.5

python test.py
```
