#! /usr/bin/python
# -*- encoding: utf-8 -*-

import pdb
import os
import glob
import argparse
import numpy
from scipy.io import wavfile
from tqdm import tqdm

from multiprocessing import Pool

parser = argparse.ArgumentParser(description="TrainArgs")

parser.add_argument('--npz_dir', type=str,
                    default="voxceleb2/dev/npz", help='')
parser.add_argument('--txt_dir', type=str,
                    default="voxceleb2/dev/txt", help='')
parser.add_argument('--wav_dir', type=str,
                    default="voxceleb2/dev/wav", help='')
parser.add_argument('--output',  type=str, default="data/dev.txt", help='')

args = parser.parse_args()

files = glob.glob(args.npz_dir+'/*/*/*.npz')

g = open(args.output, 'w')


def get(fname):
  wavname = fname.replace(args.npz_dir, args.wav_dir).replace('.npz', '.wav')
  txtname = fname.replace(args.npz_dir, args.txt_dir).replace('.npz', '.txt')

  ## Read offset
  f = open(txtname, 'r')
  txt = f.readlines()
  f.close()

  if txt[2].split()[0] == 'Offset':
    offset = txt[2].split()[2]
  else:
    print('Skipped %s - unable to read offset' % fname)
    return None

  ## Read video length
  counted_frames = len(numpy.load(fname).keys())

  if counted_frames == 0:
    print('Skipped %s - frame number inconsistent' % fname)
    return None

  ## Read audio
  sample_rate, audio = wavfile.read(wavname)

  lendiff = len(audio)/640 - counted_frames

  if abs(lendiff) > 1:
    print('Skipped %s - audio and video lengths different' % fname)
    return None

  return fname, wavname, offset, counted_frames


with Pool(56) as pool:
  ret = pool.imap(get, files)
  result = list(tqdm(ret, total=len(files)))

for r in result:
  if r is not None:
    fname, wavname, offset, counted_frames = r
    g.write('%s %s %s %d\n' % (fname, wavname, offset, counted_frames))
