import os
import warnings
import litdata as ld
from torchvision.io import read_video
import subprocess
from tqdm import tqdm
import argparse
import pickle as pkl

SELECT_TEMPORAL_INDEX=True
TEMPORAL_INDEX = 0

def safe_makedir(path):
	if not os.path.exists(path):
		os.makedirs(path)

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

def get_video_tensor(file_path):
	video, _, info = read_video(file_path, pts_unit='sec') #still THWC and 0-255
	return {"video": video}
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--pkl_path', type=str)
	parser.add_argument('--output_dir', type=str)
	parser.add_argument('--nchunks', type=int)
	parser.add_argument('--chunk_index', type=int)
	parser.add_argument('--chunk_bytes', type=str)
	args = parser.parse_args()
	print(f'Settings for optimization run: {args}')

	pkl_path = args.pkl_path
	dataset_dir = args.output_dir
	safe_makedir(dataset_dir)

	nchunks = args.nchunks
	chunk_index = args.chunk_index
	chunk_bytes = args.chunk_bytes
	num_workers=os.cpu_count()
	print(f'Reading {pkl_path}...')
	
	paths = pkl.load(open(pkl_path, 'rb'))
	nfiles = len(paths)
	print(f'Got total files: {nfiles}')
	# Filter out empty paths
	split_size = nfiles // nchunks
	video_paths = paths[chunk_index*split_size:(chunk_index+1)*split_size]

	if SELECT_TEMPORAL_INDEX:
		string_match = f'_{TEMPORAL_INDEX}.mp4'
		print(f'Filtering by temporal index, only keeping files with suffix {string_match}')
		video_paths = [p for p in video_paths if string_match in p]

	save_dir = os.path.join(dataset_dir, f'{os.path.basename(dataset_dir)}_chunk{chunk_index}')
	print(f'Saving to {save_dir}')
	safe_makedir(save_dir)
	print(f'Subset length: {len(video_paths)}')
	ld.optimize(fn=get_video_tensor, inputs=video_paths, output_dir=save_dir, 
		num_workers=os.cpu_count(), chunk_bytes=chunk_bytes, compression='zstd', keep_data_ordered=False)

