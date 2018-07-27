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

videoFlag = True
framesSorted = []
frameLanes = {}

class SCNN:

	def __init__(self):
		parser = argparse.ArgumentParser()
		parser.add_argument("source", help = "Path to video or image directory")
		parser.add_argument("-s", "--scnn", help = "RUN SCNN probability map generation", action = "store_true", default = False)
		parser.add_argument("-v", "--video", help = "RUN video generation", action = "store_true", default = False)
		parser.add_argument("-d", "--debug", help = "RUN in debug mode (output displayed)", action = "store_true", default = False)
		self.args = parser.parse_args()

		self.source = self.args.source
		self.base = "/".join(self.source.split("/")[:-1]) + "/"
		self.predict = self.base + "predicts/"
		self.destination = self.base + "Spliced"
		self.path2prob = self.base + "Prob"
		self.path2predict = self.predict + self.destination[5:] + "/"
		self.path2vid = self.base + "Videos"
		self.path2curves = self.base + "Curves"


	# check for existence of the folder first to prevent error generation
	def makedir(self, path):
		if os.path.isdir(path):
			pass
			shutil.rmtree(path)
		os.makedirs(path)


	# extend context manager's "cd"; goes back to origin directory after completion
	@contextmanager
	def cd(self, newdir):
		prevdir = os.getcwd()
		os.chdir(os.path.expanduser(newdir))
		try:
			yield
		finally:
			os.chdir(prevdir)


	# Check whether input is video or image
	def vidOrImg(self):
		global videoFlag
		if (self.source[-4:] == ".mp4"):
			videoFlag = True
			print("**** DETECTED VIDEO FILE ****")
		else:
			videoFlag = False
			print("**** DETECTED IMAGE FOLDER ****")
		self.makedir(self.destination)


	# Use ffmpeg to generate key frames from video OR generate image directory (to generate lane curves, the coordinates text
	# file and the respective frame needs to be in the same folder); therefore, a new directory is created for immutability of input. The
	# directory is called "Spliced/"
	def splice(self):
		global videoFlag
		if (videoFlag):
			print("**** SPLICING VIDEO INTO FRAMES ****")
			mp4File = open(self.source, 'r')
			os.system("ffmpeg -loglevel panic -i {0} {1}/%1d.jpg".format(mp4File.name, self.destination))
			mp4File.close()
		else:
			print("**** CREATING IMAGE DIRECTORY ****")
			for image in os.listdir(self.source):
				if image.endswith(".jpg"):
					image_path = self.source + "/" + image
					shutil.copy(image_path, self.destination)
			frames = os.listdir(self.destination)
			framesSorted = sorted(([s.strip('.jpg') for s in frames]), key = int)
			count = 0
			for frame in framesSorted:
				count = count + 1
				os.rename(self.destination + "/" + frame + ".jpg", self.destination + "/" + str(count) + ".jpg")


	# SCNN requires a text file which points to every input frame. It uses that text file to generate probability maps
	def makeTest(self):
		print("**** MAKING TEST.TXT FILE ****")
		global framesSorted
		frames = os.listdir(self.destination)
		framesSorted = sorted(([s.strip('.jpg') for s in frames]), key = int)
		txt = open(self.base + "test.txt", "w+")
		for i in range(len(framesSorted)):
			txt.write("/" + self.destination[5:] + "/" + framesSorted[i] + ".jpg" + "\n")
		txt.close()


	# Simply, run the SCNN testing script to generate a probability map for each lane per frame (4 maps per frame). The probability
	# maps are stored in the "predicts/" folder
	def probMaps(self):
		if (self.args.scnn):
			print("**** MAKING PROBABILITY MAPS ****")
			self.makedir(self.predict)
			if (self.args.debug):
				os.system("sh ~/SCNN/experiments/test.sh")
			else:
				os.system("sh ~/SCNN/experiments/test.sh >/dev/null")
		else:
			print("**** BYPASSED: SCNN PROBABILITY MAPS ****")


	# Add each lane probability map to generate per frame probability map (will all lanes in one image). Simple addition of
	# probability maps is not ideal; therefore, this block might be removed in later releases (it is present for convenience). The
	# results are stored in the "Prob/" folder
	def avgProbMaps(self):
		print("**** MAKING AVERAGES FROM PROBABILITY MAP ****")
		self.makedir(self.path2prob)
		global framesSorted
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
			im1arr = asarray(Image.open(self.path2predict + imName + "_1_avg.png"))
			im2arr = asarray(Image.open(self.path2predict + imName + "_2_avg.png"))
			im3arr = asarray(Image.open(self.path2predict + imName + "_3_avg.png"))
			im4arr = asarray(Image.open(self.path2predict + imName + "_4_avg.png"))
			#new_img = Image.blend(background, overlay, 0.5)
			#new_img = cv2.add(background, overlay)
			addition = im1arr + im2arr + im3arr + im4arr
			new_img = Image.fromarray(addition)
			new_img.save(self.path2prob + "/" + imName + ".jpg", "JPEG")


	# Run matlab script to make lane coordinates. The coordinates are stored in "[frame no.].lines.txt" in the "Spliced/" folder
	def laneCoord(self):
		print("**** MAKING LANE COORDINATES ****")
		with self.cd("~/SCNN/tools/prob2lines"):
			if (self.args.debug):
				os.system("matlab -nodisplay -r \"main;exit\"")
			else:
				os.system("matlab -nodisplay -r \"main;exit\" >/dev/null")


	# Run seg_label_generate to create lane curves by using a cubic spline. The curves are drawn on top of the input image and 
	# stored in a separate folder called "Curves/"
	def laneCurve(self):
		print("**** MAKING LANE CURVES ****")
		self.makedir(self.path2curves)
		with self.cd("~/SCNN/seg_label_generate"):
			os.system("make clean")
			if (self.args.debug):
				os.system("make")
				os.system("sh labelGen.sh")
			else:
				os.system("make >/dev/null")
				os.system("sh labelGen.sh >/dev/null")


	# Use ffmpeg to generate videos using given key frames with lane curves
	def genVideo(self):
		if (self.args.video):
			print("**** MAKING ALL VIDEOS ****")
			self.makedir(self.path2vid)
			os.system("ffmpeg -loglevel panic -framerate 24 -i {0}/%1d.jpg {1}/prob.mp4".format(self.path2prob, self.path2vid))
			os.system("ffmpeg -loglevel panic -framerate 24 -i {0}/%1d.jpg {1}/curve.mp4".format(self.path2curves, self.path2vid))
		else:
			print("**** BYPASSED: VIDEO GENERATION ****")


	# Read "[frame no.].exist.txt" to check how many lanes there are
	def checkLanes(self):
		print("**** FINDING NUMBER OF LANES ****")
		global framesSorted
		global frameLanes
		for i in range(len(framesSorted)):
			frame = str(i + 1)
			finalList = []
			existName = self.path2predict + frame + ".exist.txt"
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


	# To find out which lane the car is in:
	# 1. Read "[frame no.].lines.txt generated by the matlab code to obtain all coordinates for each frame
	# 2. Check whether a coordinate exists (for each detected lane) that is on the bottom-left quadrant
	# 3. If it does, then a lane exists that is on the left side of the car
	# Note: a key assumption is that the camera is always on the middle of the car i.e. for each frame, the car's location is in the
	# middle. This is true for 99% of the cases; however, if not, then offset xCoord by whatever value necessary
	def checkCurrentLane(self):
		print("**** FINDING CURRENT LANE ****")
		global framesSorted
		global frameLanes
		for i in range(len(framesSorted)):
			frame = str(i + 1)
			count = 0
			coordName = self.destination + "/" + frame + ".lines.txt"
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


	# Read "[frame no.].conf.txt" to check what the confidence for each lane detected is
	def conf(self):
		print("**** FINDING CONFIDENCE ****")
		global framesSorted
		global frameLanes
		for i in range(len(framesSorted)):
			frame = str(i + 1)
			confName = self.path2predict + frame + ".conf.txt"
			confTxt = open(confName, "r")
			confRead = confTxt.read().replace("\n", "")
			confList = confRead.split(" ")[:-1]
			frameLanes[int(frame)].append(confList)
			confTxt.close()


	# Generate my bruh jason 
	def json(self):
		print("**** MAKING JSON OUTPUT ****")
		jsonFile = open(self.base + "data.json", "w+")
		data = []
		global framesSorted
		global frameLanes
		for i in range(len(framesSorted)):
			frame = i + 1
			count = frameLanes[frame][0]
			lane = frameLanes[frame][1]
			conf = frameLanes[frame][2]
			data.append({"frame": frame, "lanes_count": count, "current_lane": lane, "confidence": conf})
		json.dump(data, jsonFile, indent=4)
		jsonFile.close()

	# Run the whole pipeline
	def runAll(self):
		self.vidOrImg()
		self.splice()
		self.makeTest()
		self.probMaps()
		self.avgProbMaps()
		self.laneCoord()
		self.laneCurve()
		self.genVideo()
		self.checkLanes()
		self.checkCurrentLane()
		self.conf()
		self.json()

if __name__ == "__main__":
	scnnTest =  SCNN()
	scnnTest.runAll()