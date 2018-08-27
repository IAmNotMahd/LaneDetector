# ***************
# INSTALL OPENCV
# ***************

# VERSION TO BE INSTALLED
OPENCV_VERSION='3.4.2'

# 1. KEEP UBUNTU OR DEBIAN UP TO DATE
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y dist-upgrade
sudo apt-get -y autoremove

# 2. INSTALL THE DEPENDENCIES
# Build tools:
sudo apt-get install -y build-essential cmake
# GUI (if you want to use GTK instead of Qt, replace 'qt5-default' with 'libgtkglext1-dev' and remove '-DWITH_QT=ON' option in CMake):
sudo apt-get install -y qt5-default libvtk6-dev
# Media I/O:
sudo apt-get install -y zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev libjasper-dev libopenexr-dev libgdal-dev
# Video I/O:
sudo apt-get install -y libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm libopencore-amrnb-dev libopencore-amrwb-dev libv4l-dev libxine2-dev
# Parallelism and linear algebra libraries:
sudo apt-get install -y libtbb-dev libeigen3-dev
# Python:
sudo apt-get install -y python-dev python-tk python-numpy python3-dev python3-tk python3-numpy
# Java:
sudo apt-get install -y ant default-jdk
# Documentation:
sudo apt-get install -y doxygen

# 3. INSTALL THE LIBRARY
sudo apt-get install -y unzip wget
wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
unzip ${OPENCV_VERSION}.zip
rm ${OPENCV_VERSION}.zip
mv opencv-${OPENCV_VERSION} OpenCV
cd OpenCV
mkdir build
cd build
cmake -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON -DWITH_XINE=ON -DBUILD_EXAMPLES=ON -DENABLE_PRECOMPILED_HEADERS=OFF ..
make -j4
sudo make install
sudo ldconfig

# 4. EXECUTE SOME OPENCV EXAMPLES AND COMPILE A DEMONSTRATION
# To complete this step, please visit 'http://milq.github.io/install-opencv-ubuntu-debian'.


# ***************
# INSTALL CUDA Toolkit
# ***************

# 1. Get the package from Nvidia
# http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
rm cuda-repo-ubuntu1404_7.5-18_amd64.deb

# 2. Set environment variables
echo 'export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=/usr/local/cuda
export PATH=$PATH:$CUDA_ROOT/bin:$HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64
' >> ~/.bashrc

# 3. Let terminal know of the changes to the .bashrc file
source .bashrc
sudo apt-get update 
sudo apt-get install -y cuda

# 4. Check if installation is successful by running the next line
# nvcc -V