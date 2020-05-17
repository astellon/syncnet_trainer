import glob
import os
import subprocess
import tempfile

from tqdm import tqdm
from multiprocessing import Pool

import numpy

import torch
import torchvision

from PIL import Image
from io import BytesIO

def split_video(filename, dstdir):
  dst = os.path.join(dstdir, 'image-%05d.jpg')
  cmd = ['ffmpeg', '-i', filename, "-y", "-vcodec", "mjpeg", "-r", "25", "-loglevel", "error", dst]
  ret = subprocess.run(cmd)
  if ret.returncode != 0:
    print('failed: ' + filename)

def read_image_blobs(dir):
  pathes = glob.glob(os.path.join(dir, "*.jpg"))
  images = []
  for p in pathes:
    with open(p, 'rb') as f:
      blob = numpy.fromfile(f, dtype='uint8')
      images.append(blob)
  return images

def save_frames(file_name, images):
  frames = { 'frame{}'.format(i):images[i] for i in range(len(images)) }
  numpy.savez(file_name, **frames)

def file_list(rootdir):
  rootdir = os.path.abspath(rootdir)
  ids = glob.glob(os.path.join(rootdir, '*', '*', '*.mp4'))
  return ids

def convert_to_npz(src, dst, tmpdir):
  split_video(src, tmpdir)
  images = read_image_blobs(tmpdir)
  save_frames(dst, images)

def _process(args):
  src, dst = args
  os.makedirs(os.path.dirname(dst), exist_ok=True)
  with tempfile.TemporaryDirectory() as tmp:
    convert_to_npz(src, dst, tmp)
  return None

def dstpath(src, dstdir):
  dirs = (os.path.splitext(src)[0] + '.npz').split(os.sep)
  # join speaker id, video id and file name
  dst = os.path.join(dstdir, dirs[-3], dirs[-2], dirs[-1])
  return dst

def convert_all(srcdir, dstdir):
  files = file_list(srcdir)
  srcdst = [ (src, dstpath(src, dstdir)) for src in files ]

  with Pool(56) as pool:
    ret = pool.imap(_process, srcdst)
    result = list(tqdm(ret, total=len(files)))

def load_frames(file_name, max_frames, start_frame):
  frames = numpy.load(file_name)

  images = []
  for i in range(start_frame, start_frame+max_frames):
    img = Image.open(BytesIO(frames['frame{}'.format(i)]))
    images.append(torchvision.transforms.ToTensor()(img))

  images = torch.stack(images)
  # DCHW -> CDHW
  images = images.permute((1,0,2,3)).unsqueeze(0)

  return images

if __name__ == '__main__':
  srcdir = '/gs/hs1/tga-i/goto/dataset/voxceleb/vox2/test/mp4'
  dstdir = '/gs/hs1/tga-i/goto/dataset/voxceleb/vox2/test/npz'

  convert_all(srcdir, dstdir)
