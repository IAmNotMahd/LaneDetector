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

video_flag = True
frames_sorted = []
frame_lanes = {}
num_images = 1001

class SCNN:

	def __init__(self, **kwargs):
		current_dir = os.path.dirname(__file__)
		path_2_SCNN = "/".join(current_dir.split("/")[:-1]) + "/" + "SCNN/"
		path_2_data = path_2_SCNN + "data/"
		path_2_source = path_2_data + "Source"
		path_2_source = "".join(path_2_source.rsplit(path_2_SCNN))
		os.chdir(path_2_SCNN)
		self.makedir(path_2_source)

		origin = kwargs['source']
		src_files = os.listdir(origin)
		for file_name in src_files:
			full_file_name = os.path.join(origin, file_name)
			if (os.path.isfile(full_file_name)):
				shutil.copy(full_file_name, path_2_source)

		self.source = path_2_source
		self.environ = kwargs['environ']
		self.scnn = kwargs['scnn']
		self.video = kwargs['video']
		self.debug = kwargs['debug']
		self.clean = kwargs['clean']

		if (self.environ):
			self.scnn = os.getenv("SCNN_SCNN")
			self.video = os.getenv("SCNN_VIDEO")
			self.debug = os.getenv("SCNN_DEBUG")
			self.clean = os.getenv("SCNN_CLEAN")

		self.base = "/".join(self.source.split("/")[:-1]) + "/"
		self.predict = self.base + "predicts/"
		self.destination = self.base + "Spliced"
		self.path_2_prob = self.base + "Prob"
		self.path_2_predict = self.predict + self.destination[5:] + "/"
		self.path_2_vid = self.base + "Videos"
		self.path_2_curves = self.base + "Curves"


	# optional method to check for commandline arguments and flags
	def parse(self):
		parser = argparse.ArgumentParser()
		parser.add_argument("source", help = "Path to video or image directory")
		parser.add_argument("-s", "--scnn", help = "RUN SCNN probability map generation", action = "store_true", default = False)
		parser.add_argument("-v", "--video", help = "RUN video generation", action = "store_true", default = False)
		parser.add_argument("-d", "--debug", help = "RUN in debug mode (output displayed)", action = "store_true", default = False)
		args = parser.parse_args()


	# check for existence of the folder first to prevent error generation
	def makedir(self, path):
		if os.path.isdir(path):
			pass
			shutil.rmtree(path)
		os.makedirs(path)


	# extend context manager's "cd"; goes back to origin directory after completion
	@contextmanager
	def cd(self, new_dir):
		prev_dir = os.getcwd()
		os.chdir(os.path.expanduser(new_dir))
		try:
			yield
		finally:
			os.chdir(prev_dir)


	# Check whether input is video or image
	def vid_or_img(self):
		global video_flag

		if (self.source[-4:] == ".mp4"):
			video_flag = True
			print("**** DETECTED VIDEO FILE ****")
		else:
			video_flag = False
			print("**** DETECTED IMAGE FOLDER ****")

		self.makedir(self.destination)


	# Use ffmpeg to generate key frames from video OR generate image directory (to generate lane curves, the coordinates text
	# file and the respective frame needs to be in the same folder); therefore, a new directory is created for immutability of input. The
	# directory is called "Spliced/"
	def splice(self):
		global video_flag

		if (video_flag):
			print("**** SPLICING VIDEO INTO FRAMES ****")
			mp4_file = open(self.source, 'r')
			os.system("ffmpeg -loglevel panic -i {0} -r 0.1 {1}/%1d.jpg".format(mp4_file.name, self.destination))
			mp4_file.close()

		else:
			print("**** CREATING IMAGE DIRECTORY ****")
			for image in os.listdir(self.source):
				if image.endswith(".jpg"):
					image_path = self.source + "/" + image
					shutil.copy(image_path, self.destination)

			frames = os.listdir(self.destination)
			frames_sorted = sorted(([s.strip('.jpg') for s in frames]), key = int)
			count = 0

			for frame in frames_sorted:
				count = count + 1
				os.rename(self.destination + "/" + frame + ".jpg", self.destination + "/" + str(count) + ".jpg")


	# SCNN requires a text file which points to every input frame. It uses that text file to generate probability maps
	def make_test(self):
		print("**** MAKING TEST.TXT FILE ****")
		global frames_sorted

		frames = os.listdir(self.destination)
		frames_sorted = sorted(([s.strip('.jpg') for s in frames]), key = int)
		txt = open(self.base + "test.txt", "w+")

		for i in range(len(frames_sorted)):
			txt.write("/" + self.destination[5:] + "/" + frames_sorted[i] + ".jpg" + "\n")

		txt.close()


	# Resize image to comply with SCNN requirements
	def resize(self):
		print("**** MAKING RESIZED IMAGES ****")
		global frames_sorted
		out_width = 1640
		out_height = 590

		for i in range(len(frames_sorted)):
			orig_image = Image.open(self.destination + "/" + str(i + 1) + ".jpg")
			scaled_image = orig_image.resize((out_width, out_height), Image.NEAREST)
			scaled_image.save(self.destination + "/" + str(i + 1) + ".jpg")


	# Simply, run the SCNN testing script to generate a probability map for each lane per frame (4 maps per frame). The probability
	# maps are stored in the "predicts/" folder
	def prob_maps(self):
		if (self.scnn):
			print("**** MAKING PROBABILITY MAPS ****")
			self.makedir(self.predict)

			model = "-model experiments/pretrained/model_best_rz.t7 "
			data = "-data ./data "
			val = "-val " + self.base + "test.txt "
			save = "-save " + self.predict + " "
			dataset = "-dataset laneTest "
			share_grad_input = "-shareGradInput true "
			n_threads = "-nThreads 2 "
			n_GPU = "-nGPU 1 "
			batch_size = "-batchSize 1 "
			smooth = "-smooth true "

			if (self.debug):
				os.system("rm ./gen/laneTest.t7")
				os.system("th testLane.lua " + model + data + val + save + dataset + share_grad_input + n_threads + n_GPU + batch_size + smooth)
			else:
				os.system("rm ./gen/laneTest.t7 >/dev/null")
				os.system("th testLane.lua " + model + data + val + save + dataset + share_grad_input + n_threads + n_GPU + batch_size + smooth + ">/dev/null")

		else:
			print("**** BYPASSED: SCNN PROBABILITY MAPS ****")


	# Add each lane probability map to generate per frame probability map (will all lanes in one image). Simple addition of
	# probability maps is not ideal; therefore, this block might be removed in later releases (it is present for convenience). The
	# results are stored in the "Prob/" folder
	def avg_prob_maps(self):
		print("**** MAKING AVERAGES FROM PROBABILITY MAP ****")
		self.makedir(self.path_2_prob)
		global frames_sorted

		for i in range(len(frames_sorted)):
			im_name = str(i + 1)
			im1_arr = asarray(Image.open(self.path_2_predict + im_name + "_1_avg.png"))
			im2_arr = asarray(Image.open(self.path_2_predict + im_name + "_2_avg.png"))
			im3_arr = asarray(Image.open(self.path_2_predict + im_name + "_3_avg.png"))
			im4_arr = asarray(Image.open(self.path_2_predict + im_name + "_4_avg.png"))
			#new_img = Image.blend(background, overlay, 0.5)
			#new_img = cv2.add(background, overlay)
			addition = im1_arr + im2_arr + im3_arr + im4_arr
			new_img = Image.fromarray(addition)
			new_img.save(self.path_2_prob + "/" + im_name + ".jpg", "JPEG")


	# Run matlab script to make lane coordinates. The coordinates are stored in "[frame no.].lines.txt" in the "Spliced/" folder
	def lane_coord(self):
		print("**** MAKING LANE COORDINATES ****")

		with self.cd("./tools/prob2lines"):
			path_2_SCNN = "../../"
			exp1 = "model_best_rz"
			data1 = path_2_SCNN + "data"
			prob_root1 = path_2_SCNN + self.predict + self.destination[5:]
			output1 = path_2_SCNN + self.destination
			args = "\'" + exp1 + "\', \'" + data1 + "\', \'" + prob_root1 + "\', \'" + output1 + "\'"

			if (self.debug):
				os.system("matlab -nodisplay -r \"try coords(" + args + "); catch; end; quit\"")
			else:
				os.system("matlab -nodisplay -r \"try coords(" + args + "); catch; end; quit\" >/dev/null")


	# Run seg_label_generate to create lane curves by using a cubic spline. The curves are drawn on top of the input image and 
	# stored in a separate folder called "Curves/"
	# flag details are: 
	# 	-l: image list file to process
	# 	-m: set mode to "imgLabel" or "trainList"
	# 	-d: dataset path
	# 	-w: the width of lane labels generated
	# 	-o: path to save the generated labels
	# 	-s: visualize annotation, remove this option to generate labels
	def lane_curve(self):
		print("**** MAKING LANE CURVES ****")
		self.makedir(self.path_2_curves)

		path_2_SCNN = "../"
		list_file = "-l " + path_2_SCNN + self.base + "test.txt "
		mode = "-m " + "imgLabel "
		data = "-d " + path_2_SCNN + "data "
		width = "-w " + "16 "
		output = "-o " + path_2_SCNN + "data "
		vis = "-s "

		with self.cd("./seg_label_generate"):
			os.system("make clean")
			if (self.debug):
				os.system("make")
				os.system("./seg_label_generate " + list_file + mode + data + width + output + vis)
			else:
				os.system("make >/dev/null")
				os.system("./seg_label_generate " + list_file + mode + data + width + output + vis + ">/dev/null")


	# Use ffmpeg to generate videos using given key frames with lane curves
	def gen_video(self):

		if (self.video):
			print("**** MAKING ALL VIDEOS ****")
			self.makedir(self.path_2_vid)
			os.system("ffmpeg -loglevel panic -framerate 24 -i {0}/%1d.jpg {1}/prob.mp4".format(self.path_2_prob, self.path_2_vid))
			os.system("ffmpeg -loglevel panic -framerate 24 -i {0}/%1d.jpg {1}/curve.mp4".format(self.path_2_curves, self.path_2_vid))
		else:
			print("**** BYPASSED: VIDEO GENERATION ****")


	# Read "[frame no.].exist.txt" to check how many lanes there are
	def check_lanes(self):
		print("**** FINDING NUMBER OF LANES ****")
		global frames_sorted
		global frame_lanes

		for i in range(len(frames_sorted)):
			frame = str(i + 1)
			final_list = []
			exist_name = self.path_2_predict + frame + ".exist.txt"
			exist_txt = open(exist_name, 'r')
			exist_read = exist_txt.read().replace("\n", "")
			exist_list = exist_read.split(" ")[:-1]
			lane_count = 0

			for j in exist_list:
				if j == '1':
					lane_count = lane_count + 1

			final_list.append(lane_count)
			frame_lanes[int(frame)] = final_list
			exist_txt.close()


	# To find out which lane the car is in:
	# 1. Read "[frame no.].lines.txt generated by the matlab code to obtain all coordinates for each frame
	# 2. Check whether a coordinate exists (for each detected lane) that is on the bottom-left quadrant
	# 3. If it does, then a lane exists that is on the left side of the car
	# Note: a key assumption is that the camera is always on the middle of the car i.e. for each frame, the car's location is in the
	# middle. This is true for 99% of the cases; however, if not, then offset xCoord by whatever value necessary
	def check_current_lane(self):
		print("**** FINDING CURRENT LANE ****")
		global frames_sorted
		global frame_lanes

		for i in range(len(frames_sorted)):
			frame = str(i + 1)
			count = 0
			coord_name = self.destination + "/" + frame + ".lines.txt"
			coord_txt = open(coord_name, "r")
			coord_read = coord_txt.readlines()

			for coord_line in coord_read:
				coord_line = coord_line.strip("\n")
				coord_list = coord_line.split(" ")
				coordX_list = []
				coordY_list = []
				found_lane = False

				for k in range(0, len(coord_list) - 1, 2):
					x_coord = int(coord_list[k])
					y_coord = int(coord_list[k + 1])
					coordX_list.append(x_coord)
					coordY_list.append(y_coord)

					if (found_lane == False):
						if ((x_coord < 700) and (y_coord > 350)):
							found_lane = True
							count = count + 1

			frame_lanes[int(frame)].append(count)
			coord_txt.close()


	# Read "[frame no.].conf.txt" to check what the confidence for each lane detected is
	def conf(self):
		print("**** FINDING CONFIDENCE ****")
		global frames_sorted
		global frame_lanes

		for i in range(len(frames_sorted)):
			frame = str(i + 1)
			conf_name = self.path_2_predict + frame + ".conf.txt"
			conf_txt = open(conf_name, "r")
			conf_read = conf_txt.read().replace("\n", "")
			conf_list = conf_read.split(" ")[:-1]
			frame_lanes[int(frame)].append(conf_list)
			conf_txt.close()


	# Generate my bruh jason 
	def json(self):
		print("**** MAKING JSON OUTPUT ****")
		json_file = open(self.base + "data.json", "w+")
		data = []
		global frames_sorted
		global frame_lanes

		for i in range(len(frames_sorted)):
			frame = i + 1
			count = frame_lanes[frame][0]
			lane = frame_lanes[frame][1]
			conf = frame_lanes[frame][2]
			data.append({"frame": frame, "lanes_count": count, "current_lane": lane, "confidence": conf})
		
		json.dump(data, json_file, indent=4)
		json_file.close()
		
		return json.dumps(data, indent=4)


	# Clean all temporary files if relevant flag passes
	def clean_all(self):
		if (self.clean):
			print("**** CLEANING TEMPORARY FILES AND FOLDERS ****")
			shutil.rmtree(self.path_2_curves)
			os.remove(self.base + "data.json")
			shutil.rmtree(self.predict)
			shutil.rmtree(self.path_2_prob)
			shutil.rmtree(self.source)
			shutil.rmtree(self.destination)
			os.remove(self.base + "test.txt")


	# Run the whole pipeline
	def run_all(self):
		self.vid_or_img()
		self.splice()
		self.make_test()
		self.resize()
		self.prob_maps()
		self.avg_prob_maps()
		self.lane_coord()
		self.lane_curve()
		self.gen_video()
		self.check_lanes()
		self.check_current_lane()
		self.conf()
		ret = self.json()
		self.clean_all()
		
		return ret


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("source", help = "Path to video or image directory")
	parser.add_argument("-e", "--environ", help = "USE Environment Variables instead", action = "store_true", default = False)
	parser.add_argument("-s", "--scnn", help = "RUN SCNN probability map generation", action = "store_true", default = False)
	parser.add_argument("-v", "--video", help = "RUN video generation", action = "store_true", default = False)
	parser.add_argument("-d", "--debug", help = "RUN in debug mode (output displayed)", action = "store_true", default = False)
	parser.add_argument("-c", "--clean", help = "REMOVE all generated folders and files", action = "store_true", default = False)
	args = vars(parser.parse_args())

	scnn_test = SCNN(**args)
	scnn_test.run_all()
