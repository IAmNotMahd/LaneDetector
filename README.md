## LaneDetector

### Introduction
The purpose of Lane Detector is to take Swarm Image or Video and detect upto 4 lanes in the input; it then generates a suitable output (image or video) with the lanes marked. LaneDetector, finally, tells you for each frame (in a JSON file):

1. Total number of lanes
2. Which lane the car is in

LaneDetector, at its core, is Spatial Convolutional Neural Network (SCNN) developed by Xingang Pan. SCNN itself is based on ResNet.

### Future Improvements
- Extend to 8 lanes
- Detect type of lane marking as well (doible yellow line, single white line etc.)
- Propagate information between frames
- Train SCNN more on swarm images

### Requirements
1. CUDA enabled GPU (SCNN recommends 3Gigabyte GPU for testing and 4 12Gigabyte GPUs for training)
2. CUDA toolkit 9.2 or later
3. cuDNN 7.1.4 or later
4. Torch7 (install.sh provides an option for torch)
5. OpenCV 3.3 or later
6. Matlab R2014a or later

### Installation
Run install.sh with the required flags that you need
		a. "-t" to install torch
		b. "-d" to download the dataset from CULane
		c. "-w" to download the weigths file from S3
		d. "-c" to clean up the custom code and replace with original SCNN files
install.sh copies custom scripts from the Backup folder into their correct directories

