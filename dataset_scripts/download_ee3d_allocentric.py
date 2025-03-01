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

DATASET = 'Allocentric'	
ROOT_URL = f'https://eventego3d.mpi-inf.mpg.de/{DATASET}'

EE3D_R_files = [
    'allocentric_basavaraj_001.tar.gz', 
    'allocentric_christen_001.tar.gz', 
    'allocentric_juniad_001.tar.gz', 
    'allocentric_karthick_001.tar.gz', 
    'allocentric_lalie_001.tar.gz', 
    'allocentric_neel_001.tar.gz', 
    'allocentric_oasis_001.tar.gz', 
    'allocentric_pramod_001.tar.gz', 
    'allocentric_pranay_002.tar.gz', 
    'allocentric_shreyas_001.tar.gz', 
    'allocentric_zchristen_fast_slow_001.tar.gz',
]

EE3D_W_files = [
    'allocentric_Session_1.tar.gz', 
    'allocentric_Session_2.tar.gz', 
    'allocentric_Session_3.tar.gz', 
    'allocentric_Session_4.tar.gz', 
    'allocentric_Session_5.tar.gz', 
    'allocentric_Session_6.tar.gz', 
    'allocentric_Session_7.tar.gz', 
    'allocentric_Session_8.tar.gz', 
    'allocentric_Session_9.tar.gz', 

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
    parser = argparse.ArgumentParser(description='Download and extract EE3D-R.')
    parser.add_argument('--type', type=str, required=True, help='Type of dataset to download. Choose from [EE3D-R, EE3D-W]')
    parser.add_argument('--location', type=str, required=True, help='Location to save the downloaded and extracted files')

    args = parser.parse_args()

    assert args.type in ['EE3D-R', 'EE3D-W'], 'Type should be either EE3D-R or EE3D-W'

    save_location = os.path.join(args.location , DATASET, args.type)
    os.makedirs(save_location, exist_ok=True)

    files = EE3D_R_files if args.type == 'EE3D-R' else EE3D_W_files

    print('Downloading sequences..')
    for idx, file in enumerate(files, 1):  
        print('[{}/{}]: [{}]'.format(idx, len(files), file))
        download_and_extract(url=f'{ROOT_URL}/{file}', save_location=save_location)

    print(f"Files downloaded and extracted successfully to {save_location}")
