# Feature-Based Stereo Visual Odometry using Harris, SIFT, and BRISK Feature Detectors
This repo contains a stereo VO implementation from scratch using OpenCV implementations of Harris, SIFT, and BRISK.

## Getting Started
### Environment
1. Python>=3.6  
2. Install additional packages
    ```
    pip install -r requirements.txt
    ```
3. If desired to run the code in a docker environment, install docker. Instructions for Linux (Ubuntu): https://docs.docker.com/engine/install/ubuntu/
### Prepare Dataset
This repo uses the Canadian Planetary Emulation Terrain Energy-Aware Rover Navigation Dataset available at https://starslab.ca/enav-planetary-dataset/.
1. Set up input and output folders
    ```
    mkdir input
    mkdir output
    ```
2. Download the rover frame transforms file, camera intrinsics file, and human-readable base data folder for Run #1 (note: this folder is 16.8 GB)
    ```
    cd ./input
    wget ftp://128.100.201.179/2019-enav-planetary/rover_transforms.txt
    wget ftp://128.100.201.179/2019-enav-planetary/cameras_intrinsics.txt
    wget ftp://128.100.201.179/2019-enav-planetary/run_1/new_data/run1_base_hr.tar.gz
    ```
3. Extract data folder
    ```
    tar -xzf run1_base_hr.tar.gz
    ```
### Run Visual Odometry
The default settings run the visual odometry on the first ~80 seconds of Run #1. All outputs will be automatically saved to the output folder.
1. To run inside a docker container, run the following command in the main directory of the repo:
    ```
    docker-compose up
    ```
2. Instead, to run manually, run the following command in the main directory of the repo:
    ```
    python3 ./src/rob501_project.py --input_dir=./input --output_dir=./output
    ```
