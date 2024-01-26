#!/bin/bash
#uvicorn fastapiapp:app --host 0.0.0.0 --port 8000

/home/ubuntu/miniconda3/condabin/conda activate ds-env
/home/ubuntu/miniconda3/envs/ds-env/bin/uvicorn fastapiapp:app --host 0.0.0.0 --port 8000