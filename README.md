# Installation

```
module load anaconda3 gcc cuda
cd lib
make
```


# Execution

get an interactive session:

```
sinteractive -t 10 --gres=gpu:teslap100:1 --mem-per-cpu=2000
module load anaconda3 cuda

python test.py
```
