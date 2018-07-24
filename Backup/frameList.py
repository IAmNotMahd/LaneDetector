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

def makedir(path):
	if os.path.isdir(path):
		shutil.rmtree(path)
	os.makedirs(path)

@contextmanager
def cd(newdir):
	prevdir = os.getcwd()
	os.chdir(os.path.expanduser(newdir))
	try:
		yield
	finally:
		os.chdir(prevdir)


# ****************
# MODULE 1
# ****************
if (source[-4:] == ".mp4"):
	videoFlag = True
	print("**** DETECTED VIDEO FILE ****")
else:
	videoFlag = False
	print("**** DETECTED IMAGE FOLDER ****")
makedir(destination)


# ****************
# MODULE 2
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
# MODULE 3
# ****************
print("**** MAKING TEST.TXT FILE ****")
frames = os.listdir(destination)
framesSorted = sorted(([s.strip('.jpg') for s in frames]), key = int)
txt = open(base + "test.txt", "w+")
for i in range(len(framesSorted)):
	txt.write("/" + destination[5:] + "/" + framesSorted[i] + ".jpg" + "\n")
txt.close()


# ****************
# MODULE 4
# ****************
if (args.scnn):
	print("**** MAKING PROBABILITY MAPS ****")
	makedir(predict)
	os.system("sh ~/SCNN/experiments/test.sh")
else:
	print("**** BYPASSED: SCNN PROBABILITY MAPS ****")


# ****************
# MODULE 5
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
# MODULE 6
# ****************
print("**** MAKING LANE COORDINATES ****")
with cd("~/SCNN/tools/prob2lines"):
	pass
	os.system("matlab -nodisplay -r \"main;exit\" >/dev/null")


# ****************
# MODULE 7
# ****************
print("**** MAKING LANE CURVES ****")
makedir(path2curves)
with cd("~/SCNN/seg_label_generate"):
	os.system("make clean")
	os.system("make >/dev/null")
	os.system("sh labelGen.sh >/dev/null")


# ****************
# MODULE 8
# ****************
if (args.video):
	print("**** MAKING ALL VIDEOS ****")
	makedir(path2vid)
	os.system("ffmpeg -loglevel panic -framerate 24 -i {0}/%1d.jpg {1}/prob.mp4".format(path2prob, path2vid))
	os.system("ffmpeg -loglevel panic -framerate 24 -i {0}/%1d.jpg {1}/curve.mp4".format(path2curves, path2vid))
else:
	print("**** BYPASSED: VIDEO GENERATION ****")


# ****************
# MODULE 9
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
# MODULE 10
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
# MODULE 11
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
# MODULE 12
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