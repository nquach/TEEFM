import os
import numpy as np
import pydicom as dcm
from pydicom.pixel_data_handlers import convert_color_space
import cv2
from cv2 import imwrite
from tqdm import tqdm
import pickle as pkl
import multiprocessing
import imageio as iio 
import argparse

import litdata as ld
import torch

FRAME_SIZE = 224
TEMPORAL_DOWNSAMPLE = True #DEFAULT STRIDE is 2
GOAL_FRAMES = 64

num_cpu = multiprocessing.cpu_count()

def safe_makedir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def absoluteFilePaths(directory):
	paths = []
	for dirpath,_,filenames in os.walk(directory):
		for f in filenames:
			paths.append(os.path.abspath(os.path.join(dirpath, f)))
	return paths

def dcm2tensor(dcm_path, verbose=False):
	try:
		ds = dcm.dcmread(dcm_path)
		nparr = ds.pixel_array
		nparr_shape = nparr.shape
		if len(nparr_shape) == 4:
			if dcm.pixel_data_handlers.numpy_handler.should_change_PhotometricInterpretation_to_RGB(ds):
				nparr_rgb = convert_color_space(nparr, ds.PhotometricIntepretation, 'RGB') 
			else:
				nparr_rgb = nparr
			nframes = nparr_shape[0]
			resized_frames = []
			for i in range(nframes):
				resized_frames.append(cv2.resize(np.squeeze(nparr_rgb[i,:,:,:]), (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA))

			padded_resized_frames = []
			n_copies = int(np.ceil(GOAL_FRAMES/float(nframes)))
			for i in range(n_copies-1):
				padded_resized_frames.extend(resized_frames.copy()) #if video is too short it repeats
			video_np = np.asarray(padded_resized_frames)
			video_np = (video_np - video_np.min())/(video_np.max() - video_np.min())
			video_uint8 = (video_np * 255).round().astype(np.uint8)
			video = torch.Tensor(video_uint8) # (should be T H W C already)
			return {'video': video}
	except:
		pass

def optimize_dataset(dcm_paths, dataset_dir, num_workers=8, chunk_bytes='512MB', compression='zstd'):
	ld.optimize(fn=dcm2tensor, inputs=dcm_paths, output_dir=dataset_dir, 
		num_workers=num_workers, chunk_bytes=chunk_bytes, compression=compression)

def optimize_job(paths, save_dir, datasets_prefix, nchunks=10, chunk_index=0, verbose=False,
				num_workers=8, chunk_bytes='256MB', compression='zstd'):
	safe_makedir(save_dir)
	dataset_dir = os.path.join(save_dir, f'{datasets_prefix}_chunk{chunk_index}')
	safe_makedir(dataset_dir)
	nfiles = len(paths)
	split_size = nfiles // nchunks
	dcm_paths = [paths[i] for i in range(chunk_index*split_size, (chunk_index+1)*split_size)]
	optimize_dataset(dcm_paths, dataset_dir, num_workers=num_workers, chunk_bytes=chunk_bytes, compression=compression)
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--nchunks', type=int)
	parser.add_argument('--chunk_index', type=int)
	args = parser.parse_args()
	print(args)
	dcm_dir = '/share/pi/krhee/nquach/TEE_foundation/echo_dicom_2023-2024'
	paths_dir = '/share/pi/krhee/nquach/TEE_foundation/paths_pkls'
	safe_makedir(paths_dir)
	paths_path = os.path.join(paths_dir, os.path.basename(dcm_dir) + '.pkl')
	paths = []
	if os.path.exists(paths_path):
		print(f'Found path file: {paths_path}')
		paths = pkl.load(open(paths_path, 'rb'))
		print('Found file paths:', len(paths))
	else:
		print('No available paths file found! Calculating all available file paths...')
		paths = absoluteFilePaths(dcm_dir)
		print('Found file paths:', len(paths))
		pkl.dump(paths, open(paths_path, 'wb'))
		print(f'Saved path file to {paths_path}')

	save_dir = '/share/pi/krhee/nquach/TEE_foundation/dcm2tensor_optimized'
	datasets_prefix = 'tee-2023-2024'
	optimize_job(paths, save_dir, datasets_prefix, nchunks=args.nchunks, chunk_index=args.chunk_index, verbose=True,
				num_workers=20, chunk_bytes='512MB', compression='zstd')


