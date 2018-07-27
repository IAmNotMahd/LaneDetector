#!/usr/bin/python
import argparse
# from lane_detector import frameList
import frameList
parser = argparse.ArgumentParser()
parser.add_argument("source", help = "Path to video or image directory")
parser.add_argument("-e", "--environ", help = "USE Environment Variables instead", action = "store_true", default = False)
parser.add_argument("-s", "--scnn", help = "RUN SCNN probability map generation", action = "store_true", default = False)
parser.add_argument("-v", "--video", help = "RUN video generation", action = "store_true", default = False)
parser.add_argument("-d", "--debug", help = "RUN in debug mode (output displayed)", action = "store_true", default = False)
args = vars(parser.parse_args())

scnnTest = frameList.SCNN(**args)
scnnRun = scnnTest.runAll()