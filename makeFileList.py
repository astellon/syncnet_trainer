#! /usr/bin/python
# -*- encoding: utf-8 -*-

import pdb, os, glob, argparse, cv2
from scipy.io import wavfile
from tqdm import tqdm

from multiprocessing import Pool

parser = argparse.ArgumentParser(description = "TrainArgs")

parser.add_argument('--mp4_dir', type=str, default="voxceleb2/dev/mp4", help='')
parser.add_argument('--txt_dir', type=str, default="voxceleb2/dev/txt", help='')
parser.add_argument('--wav_dir', type=str, default="voxceleb2/dev/wav", help='')
parser.add_argument('--output',  type=str, default="data/dev.txt", help='')

args = parser.parse_args()

files = glob.glob(args.mp4_dir+'/*/*/*.mp4')

g = open(args.output,'w')

def get(fname):
  wavname = fname.replace(args.mp4_dir,args.wav_dir).replace('.mp4','.wav')
  txtname = fname.replace(args.mp4_dir,args.txt_dir).replace('.mp4','.txt')

  ## Read offset
  f = open(txtname,'r')
  txt = f.readlines()
  f.close()

  if txt[2].split()[0] == 'Offset':
    offset = txt[2].split()[2]
  else:
    print('Skipped %s - unable to read offset'%fname)
    return None

  ## Read video length
  cap = cv2.VideoCapture(fname)
  counted_frames = 0
  while True:
    ret, image = cap.read()
    if ret == 0:
      break
    else:
      counted_frames += 1
  total_frames = cap.get(7)
  cap.release()

  if total_frames != counted_frames:
    print('Skipped %s - frame number inconsistent'%fname)
    return None

  ## Read audio
  sample_rate, audio  = wavfile.read(wavname)

  lendiff = len(audio)/640 - counted_frames

  if abs(lendiff) > 1:
    print('Skipped %s - audio and video lengths different'%fname)
    return None

  return fname, wavname, offset, counted_frames

with Pool(56) as pool:
  ret = pool.imap(get, files)
  result = list(tqdm(ret, total=len(files)))

for r in result:
  if r is not None:
    fname, wavname, offset, counted_frames = r
    g.write('%s %s %s %d\n'%(fname,wavname,offset,counted_frames))
