import os
import numpy as np
import pydicom as dcm
from pydicom.pixel_data_handlers import convert_color_space
import cv2
from tqdm import tqdm
import pickle as pkl
import multiprocessing
import imageio as iio 
import argparse
import subprocess

def safe_makedir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def absoluteFilePaths(directory):
	paths = []
	for dirpath,_,filenames in os.walk(directory):
		for f in filenames:
			paths.append(os.path.abspath(os.path.join(dirpath, f)))
	return paths

def is_mp4_corrupted(filepath):
    command = [
        'ffmpeg',
        '-v', 'error',  # Only show error messages
        '-i', filepath,
        '-f', 'null',   # Output to null device
        '-'             # Read from stdin (not strictly necessary here, but common)
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        # If check=True, a CalledProcessError is raised for non-zero exit codes.
        # If no error, the file is likely not corrupted.
        return False
    except subprocess.CalledProcessError as e:
        # A non-zero exit code indicates an error during processing.
        # You can inspect e.stderr for specific error messages if needed.
        print(f"Error processing {filepath}: {e.stderr}")
        return True
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please ensure it's installed and in your PATH.")
        return True

def dcm2mp4(dcm_path, mp4_dir, new_id, 
	min_frames=64, frame_size=224, temporal_stride=4, verbose=False, recalculate=False, max_attempts=3, depth=0):
	if depth > max_attempts:
		print(f'ERROR dcm2mp4 has been called more than {max_attempts} for file {dcm_path}, skipping!')
		return
	last_mp4_path = os.path.join(mp4_dir, f'{new_id}_{temporal_stride-1}.mp4')
	if os.path.exists(last_mp4_path) and not recalculate:
		print(f'{last_mp4_path} exists! skipping {dcm_path}...')
		return

	ds = dcm.dcmread(dcm_path)
	nparr = ds.pixel_array
	nparr_shape = nparr.shape
	nframes = nparr_shape[0]
	if verbose:
		print(f'Pixel data obtained, of shape: {nparr.shape}')

	if len(nparr_shape) == 4:
		if dcm.pixel_data_handlers.numpy_handler.should_change_PhotometricInterpretation_to_RGB(ds):
			nparr_rgb = convert_color_space(nparr, ds.PhotometricIntepretation, 'RGB') 
		else:
			nparr_rgb = nparr

		resized_frames = []
		for i in range(nframes):
			resized_frames.append(cv2.resize(np.squeeze(nparr_rgb[i,:,:,:]), (frame_size, frame_size), interpolation=cv2.INTER_AREA))

		padded_resized_frames = []
		if nframes < min_frames:
			n_copies = int(np.ceil(min_frames/float(nframes)))
			for i in range(n_copies):
				padded_resized_frames.extend(resized_frames.copy())
			print(f'ncopies={n_copies}, padded_resized_frames: {len(padded_resized_frames)}')
			resized_frames = padded_resized_frames
			if verbose:
				print(f'Padded to meet minimum frames {min_frames}: now is of length {len(resized_frames)}')

		resized_frames = np.asarray(resized_frames)
		for temporal_index in range(temporal_stride):
			original_path = dcm_path
			mp4_path = os.path.join(mp4_dir, f'{new_id}_{temporal_index}.mp4')
			video = resized_frames[temporal_index::temporal_stride] #(T, H, W, C)
			writer = iio.get_writer(mp4_path, fps=10) #set FPS to fix 10 FPS
			for i in range(video.shape[0]):
				writer.append_data(np.squeeze(video[i,...]))
			writer.close()
			if verbose:
				print(f'Saved video of shape {video.shape}: {mp4_path}')
			if is_mp4_corrupted(mp4_path):
				if verbose:
					print('Error in MP4 validation: Retrying to make mp4...')
					dcm2mp4(dcm_path, mp4_dir, new_id, 
						min_frames=min_frames, frame_size=frame_size, temporal_stride=temporal_stride, 
						verbose=verbose, max_attempts=max_attempts, depth=(depth+1))
	else:
		if verbose:
			print(f'ERROR: Pixel array is not 4D: {dcm_path} is of shape {nparr_shape}')	
		return


def process_all(paths, mp4_dir, nchunks=10, chunk_index=0, 
	min_frames=64, frame_size=224, temporal_stride=4,
	verbose=False, multiprocess=True, recalculate=False):
	safe_makedir(mp4_dir)
	p = multiprocessing.Pool()
	nfiles = len(paths)
	split_size = nfiles // nchunks
	for i in range(chunk_index*split_size, (chunk_index+1)*split_size):
		new_id = str(i)
		dcm_path = paths[i]
		if multiprocess:
			res = p.apply_async(dcm2mp4, args=(dcm_path, mp4_dir, new_id, min_frames, frame_size, 
				temporal_stride, verbose, recalculate))
		else:
			dcm2mp4(dcm_path, mp4_dir, new_id,  min_frames, frame_size, 
				temporal_stride, verbose, recalculate)
	p.close()
	p.join()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--nchunks', type=int)
	parser.add_argument('--chunk_index', type=int)
	parser.add_argument('--min_frames', type=int, default=64)
	parser.add_argument('--frame_size', type=int, default=224)
	parser.add_argument('--temporal_stride', type=int, default=4)
	parser.add_argument('--verbose', action='store_true')
	parser.add_argument('--multiprocess', action='store_true')
	parser.add_argument('--recalculate', action='store_true')
	args = parser.parse_args()
	print(args)
	dcm_dir = '/home/nquach/PI_HOME/echo_dicom_2014-2023'
	paths_dir = '/home/nquach/PI_HOME/nquach/TEE_foundation/paths_pkls'
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

	mp4_dir = '/home/nquach/PI_HOME/nquach/TEE_foundation/mp4-2014-2023_v2/mp4'
	process_all(paths, mp4_dir, nchunks=args.nchunks, chunk_index=args.chunk_index, 
		min_frames=args.min_frames, frame_size=args.frame_size, temporal_stride=args.temporal_stride,
		verbose=args.verbose, multiprocess=args.multiprocess, recalculate=args.recalculate)


