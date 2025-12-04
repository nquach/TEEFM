import os
import litdata as ld
from torchvision.io import read_video
import subprocess

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
	video, _, info = read_video(file_path, pts_unit='sec')
	return {"path": file_path, "video": video}


def optimize_dataset(csv_path, dataset_dir, num_workers=8, chunk_bytes='64MB'):
 	with open(csv_file, 'r') as f:
            # Read all lines and strip whitespace
            video_paths = [line.strip() for line in f.readlines() if line.strip()]
        
	# Filter out empty paths
	video_paths = [path for path in video_paths if path]
	video_paths = [path for path in video_paths if not is_mp4_corrupted(path)] #filter out corrupted mp4 files
	ld.optimize(fn=get_video_tensor, inputs=video_paths, output_dir=dataset_dir, 
		num_workers=num_workers, chunk_bytes=chunk_bytes)

if __name == '__main__':
	csv_path1 = '/teamspace/studios/this_studio/TEEFM/mp4_paths.csv'
	csv_path2 = '/teamspace/studios/this_studio/TEEFM/val500_2023-2024.csv'
	output_dir1 = '/teamspace/studios/this_studio/opt_mp4-2014-2023'
	output_dir2 = '/teamspace/studios/this_studio/opt_mp4-2023-2024'
	num_workers = 8
	chunk_bytes = '64MB'
	optimize_dataset(csv_path1, output_dir1, num_workers, chunk_bytes)

