#!/usr/bin/python

import os
import sys
import shutil
import json
import subprocess
import argparse
from contextlib import contextmanager
from PIL import Image
from numpy import *

# Parse arguments and generate flags to bypass time-extensive code blocks
parser = argparse.ArgumentParser()
parser.add_argument("source", help = "Path to video or image directory")
parser.add_argument("-s", "--scnn", help = "RUN SCNN probability map generation", action = "store_true")
parser.add_argument("-v", "--video", help = "RUN video generation", action = "store_true")
args = parser.parse_args()

source = args.source
#source = "data/example-swarm-data/culane data"
#source = "data/example-swarm-data/swarm-data.mp4"
base = "/".join(source.split("/")[:-1]) + "/"
predict = base + "predicts/"
destination = base + "Spliced"
path2prob = base + "Prob"
path2predict = predict + destination[5:] + "/"
path2vid = base + "Videos"
path2curves = base + "Curves"
numImages = 1001

# function checks for existence of the folder first to prevent error generation
def makedir(path):
	if os.path.isdir(path):
		shutil.rmtree(path)
	os.makedirs(path)

# function extends context manager's "cd"; goes back to origin directory after completion
@contextmanager
def cd(newdir):
	prevdir = os.getcwd()
	os.chdir(os.path.expanduser(newdir))
	try:
		yield
	finally:
		os.chdir(prevdir)


# ****************
# Block 1: Check whether input is video or image
# ****************
if (source[-4:] == ".mp4"):
	videoFlag = True
	print("**** DETECTED VIDEO FILE ****")
else:
	videoFlag = False
	print("**** DETECTED IMAGE FOLDER ****")
makedir(destination)


# ****************
# Block 2: Use ffmpeg to generate key frames from video OR generate image directory (to generate lane curves, the coordinates text
# file and the respective frame needs to be in the same folder); therefore, a new directory is created for immutability of input. The
# directory is called "Spliced/"
# ****************
if (videoFlag):
	print("**** SPLICING VIDEO INTO FRAMES ****")
	mp4File = open(source, 'r')
	os.system("ffmpeg -loglevel panic -i {0} {1}/%1d.jpg".format(mp4File.name, destination))
	mp4File.close()
else:
	print("**** CREATING IMAGE DIRECTORY ****")
	for image in os.listdir(source):
		if image.endswith(".jpg"):
			image_path = source + "/" + image
			shutil.copy(image_path, destination)
	frames = os.listdir(destination)
	framesSorted = sorted(([s.strip('.jpg') for s in frames]), key = int)
	count = 0
	for frame in framesSorted:
		count = count + 1
		os.rename(destination + "/" + frame + ".jpg", destination + "/" + str(count) + ".jpg")


# ****************
# Block 3: SCNN requires a text file which points to every input frame. It uses that text file to generate probability maps
# ****************
print("**** MAKING TEST.TXT FILE ****")
frames = os.listdir(destination)
framesSorted = sorted(([s.strip('.jpg') for s in frames]), key = int)
txt = open(base + "test.txt", "w+")
for i in range(len(framesSorted)):
	txt.write("/" + destination[5:] + "/" + framesSorted[i] + ".jpg" + "\n")
txt.close()


# ****************
# Block 4: Simply, run the SCNN testing script to generate a probability map for each lane per frame (4 maps per frame). The probability
# maps are stored in the "predicts/" folder
# ****************
if (args.scnn):
	print("**** MAKING PROBABILITY MAPS ****")
	makedir(predict)
	os.system("sh ~/SCNN/experiments/test.sh")
else:
	print("**** BYPASSED: SCNN PROBABILITY MAPS ****")


# ****************
# Block 5: Add each lane probability map to generate per frame probability map (will all lanes in one image). Simple addition of
# probability maps is not ideal; therefore, this block might be removed in later releases (it is present for convenience). The
# results are stored in the "Prob/" folder
# ****************
print("**** MAKING AVERAGES FROM PROBABILITY MAP ****")
makedir(path2prob)
for i in range(len(framesSorted)):
	imName = str(i + 1)
	'''num = i * 30
	if (num < 10):
		imName = "0000" + str(num)
	elif (num < 100):
		imName = "000" + str(num)
	elif (num < 1000):
		imName = "00" + str(num)
	else:
		imName = "0" + str(num)'''
	im1arr = asarray(Image.open(path2predict + imName + "_1_avg.png"))
	im2arr = asarray(Image.open(path2predict + imName + "_2_avg.png"))
	im3arr = asarray(Image.open(path2predict + imName + "_3_avg.png"))
	im4arr = asarray(Image.open(path2predict + imName + "_4_avg.png"))
	#new_img = Image.blend(background, overlay, 0.5)
	#new_img = cv2.add(background, overlay)
	addition = im1arr + im2arr + im3arr + im4arr
	new_img = Image.fromarray(addition)
	new_img.save(path2prob + "/" + imName + ".jpg", "JPEG")


# ****************
# Block 6: Run matlab script to make lane coordinates. The coordinates are stored in "[frame no.].lines.txt" in the "Spliced/" folder
# ****************
print("**** MAKING LANE COORDINATES ****")
with cd("~/SCNN/tools/prob2lines"):
	pass
	os.system("matlab -nodisplay -r \"main;exit\" >/dev/null")


# ****************
# Block 7: Run seg_label_generate to create lane curves by using a cubic spline. The curves are drawn on top of the input image and 
# stored in a separate folder called "Curves/"
# ****************
print("**** MAKING LANE CURVES ****")
makedir(path2curves)
with cd("~/SCNN/seg_label_generate"):
	os.system("make clean")
	os.system("make >/dev/null")
	os.system("sh labelGen.sh >/dev/null")


# ****************
# Block 8: Use ffmpeg to generate videos using given key frames with lane curves
# ****************
if (args.video):
	print("**** MAKING ALL VIDEOS ****")
	makedir(path2vid)
	os.system("ffmpeg -loglevel panic -framerate 24 -i {0}/%1d.jpg {1}/prob.mp4".format(path2prob, path2vid))
	os.system("ffmpeg -loglevel panic -framerate 24 -i {0}/%1d.jpg {1}/curve.mp4".format(path2curves, path2vid))
else:
	print("**** BYPASSED: VIDEO GENERATION ****")


# ****************
# Block 9: Read "[frame no.].exist.txt" to check how many lanes there are
# ****************
print("**** FINDING NUMBER OF LANES ****")
frameLanes = {}
for i in range(len(framesSorted)):
	frame = str(i + 1)
	finalList = []
	existName = path2predict + frame + ".exist.txt"
	existTxt = open(existName, 'r')
	existRead = existTxt.read().replace("\n", "")
	existList = existRead.split(" ")[:-1]
	laneCount = 0
	for j in existList:
		if j == '1':
			laneCount = laneCount + 1
	finalList.append(laneCount)
	frameLanes[int(frame)] = finalList
	existTxt.close()


# ****************
# Block 10: To find out which lane the car is in:
# 1. Read "[frame no.].lines.txt generated by the matlab code to obtain all coordinates for each frame
# 2. Check whether a coordinate exists (for each detected lane) that is on the bottom-left quadrant
# 3. If it does, then a lane exists that is on the left side of the car
# Note: a key assumption is that the camera is always on the middle of the car i.e. for each frame, the car's location is in the
# middle. This is true for 99% of the cases; however, if not, then offset xCoord by whatever value necessary
# ****************
print("**** FINDING CURRENT LANE ****")
for i in range(len(framesSorted)):
	frame = str(i + 1)
	count = 0
	coordName = destination + "/" + frame + ".lines.txt"
	coordTxt = open(coordName, "r")
	coordRead = coordTxt.readlines()
	for coordLine in coordRead:
		coordLine = coordLine.strip("\n")
		coordList = coordLine.split(" ")
		coordXList = []
		coordYList = []
		foundLane = False
		for k in range(0, len(coordList) - 1, 2):
			xCoord = int(coordList[k])
			yCoord = int(coordList[k + 1])
			coordXList.append(xCoord)
			coordYList.append(yCoord)
			if foundLane == False:
				if ((xCoord < 700) and (yCoord > 350)):
					foundLane = True
					count = count + 1
	frameLanes[int(frame)].append(count)
	coordTxt.close()


# ****************
# Block 11: Read "[frame no.].conf.txt" to check what the confidence for each lane detected is
# ****************
print("**** FINDING CONFIDENCE ****")
for i in range(len(framesSorted)):
	frame = str(i + 1)
	confName = path2predict + frame + ".conf.txt"
	confTxt = open(confName, "r")
	confRead = confTxt.read().replace("\n", "")
	confList = confRead.split(" ")[:-1]
	frameLanes[int(frame)].append(confList)
	confTxt.close()


# ****************
# Block 12: Generate my bruh jason 
# ****************
print("**** MAKING JSON OUTPUT ****")
jsonFile = open(base + "data.json", "w+")
data = []
for i in range(len(framesSorted)):
	frame = i + 1
	count = frameLanes[frame][0]
	lane = frameLanes[frame][1]
	conf = frameLanes[frame][2]
	data.append({"frame": frame, "lanes_count": count, "current_lane": lane, "confidence": conf})
json.dump(data, jsonFile, indent=4)
jsonFile.close()
