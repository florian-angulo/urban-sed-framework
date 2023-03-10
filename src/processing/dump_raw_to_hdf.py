#!/usr/bin/env python3
import argparse
import librosa
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
import soundfile as sf
import h5py
import os
import multiprocessing as mp


parser = argparse.ArgumentParser()
parser.add_argument("input_csv")
parser.add_argument("-o", "--output", type=str, required=True, help="Output data hdf5")
parser.add_argument(
    "-g", "--group", required=True, type=str, help="Group to create in the hdf5 file"
)
parser.add_argument(
    "-f", "--folder", type=str, required=True, help="Path to audio folder"
)
parser.add_argument("-sr", type=int, default=32000)
parser.add_argument("-sep", default=",", type=str)
args = parser.parse_args()


@logger.catch
def worker(idx_filename, q):
    idx = idx_filename[0]
    fname = idx_filename[1]
    y, sr = sf.read(args.folder + fname, dtype="float32")
    if y.ndim > 1:
        # Merge channels
        y = y.mean(-1)
    if sr != args.sr:
        y = librosa.resample(y, sr, args.sr)
    res = (idx, y)
    q.put(res)
    return res


@logger.catch
def listener(q):
    """listens for messages on the q, writes to file."""
    if os.path.exists(args.output):
        hf = h5py.File(args.output, "r+")
    else:
        hf = h5py.File(args.output, "w")

    dname = f"{args.group}/audio_32k"
    if dname in hf:
        dset = hf[dname]
    else:
        dset = hf.create_dataset(dname, shape=(len(df), args.sr * 10))

    while 1:
        m = q.get()
        if m == "kill":
            logger.success("Killing listener")
            hf.close()
            break
        idx = m[0]
        audio = m[1]
        dset[idx] = audio


logger.add("somefile.log", enqueue=True)
df = pd.read_csv(args.input_csv, sep=args.sep)
assert "filename" in df.columns, "Header needs to contain 'filename'"

filenames = df["filename"].to_list()

# must use Manager queue here, or will not work
manager = mp.Manager()
q = manager.Queue()
pool = mp.Pool(mp.cpu_count() + 2)

# put listener to work first
watcher = pool.apply_async(listener, (q,))

# fire off workers
jobs = []
for i, fname in enumerate(filenames):
    job = pool.apply_async(worker, ((i, fname), q))
    jobs.append(job)

# collect results from the workers through the pool result queue
for job in tqdm(jobs):
    job.get()

# now we are done, kill the listener
q.put("kill")
pool.close()
pool.join()
