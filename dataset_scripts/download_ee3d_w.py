"""
EventEgo3D: 3D Human Motion Capture from Egocentric Event Streams
https://4dqv.mpi-inf.mpg.de/EventEgo3D/

Dataset download script
"""

import os
import argparse
import tarfile
import urllib.request
from tqdm import tqdm

DATASET = 'EE3D-W'	
ROOT_URL = f'https://eventego3d.mpi-inf.mpg.de/{DATASET}'

files = [
    'test.txt',
    'train.txt',
    'val.txt',
    'Session_1.tar.gz',
    'Session_2.tar.gz',
    'Session_3.tar.gz',
    'Session_4.tar.gz',
    'Session_5.tar.gz',
    'Session_6.tar.gz',
    'Session_7.tar.gz',
    'Session_8.tar.gz',
    'Session_9.tar.gz',
]


def extract_tar(file_path):
    # Extract the tar.gz file with progress bar
    with tarfile.open(file_path, 'r:gz') as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), desc='Extracting', unit='file') as tqdm_instance:
            for member in members:
                tar.extract(member, path=save_location)
                tqdm_instance.update(1)
    
    # Remove the tar.gz file after extraction
    os.remove(file_path)

def download_and_extract(url, save_location):
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    
    # Download the file
    file_path = os.path.join(save_location, url.split('/')[-1])
    with tqdm(unit='B', unit_scale=True, desc='Downloading '+ url.split('/')[-1]) as tqdm_instance:
        urllib.request.urlretrieve(url, filename=file_path, reporthook=lambda block_num, block_size, total_size: tqdm_instance.update(block_size))
    
    if file_path.endswith('.tar.gz'):
        extract_tar(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and extract EE3D-W.')
    parser.add_argument('--location', type=str, required=True, help='Location to save the downloaded and extracted files')

    args = parser.parse_args()

    save_location = os.path.join(args.location , DATASET)
    os.makedirs(save_location, exist_ok=True)

    print('Downloading sequences..')
    for idx, file in enumerate(files, 1):  
        print('[{}/{}]: [{}]'.format(idx, len(files), file))
        download_and_extract(url=f'{ROOT_URL}/{file}', save_location=save_location)

    print(f"Files downloaded and extracted successfully to {save_location}")
