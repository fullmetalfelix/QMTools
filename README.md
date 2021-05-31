# Installation

```
module load anaconda3 gcc cuda
cd lib
make
```


# Execution

get an interactive session:

```
sinteractive -t 10 --gres-gpu:...
module load anaconda3 cuda

python test.py
```
