'''
Lane Detector:
A tool that wraps around SCNN's testing process.
'''
import os
import sys
import shutil
import json
import subprocess
import argparse
from contextlib import contextmanager
from PIL import Image
import numpy
import time

VIDEO_FLAG = True
FRAMES_SORTED = []
FRAME_LANES = {}
NUM_IMAGES = 1001


class SCNN:
    '''
    Class is essentially an SCNN wrapper
    '''
    def __init__(self, **kwargs):
        global VIDEO_FLAG

        current_dir = os.path.dirname(os.path.abspath(__file__))
        path_2_scnn = "/".join(current_dir.split("/")[:-1]) + "/" + "SCNN/"
        path_2_data = path_2_scnn + "data/"
        path_2_source = path_2_data + "Source"
        path_2_source = "".join(path_2_source.rsplit(path_2_scnn))
        os.chdir(path_2_scnn)
        self.makedir(path_2_source)

        origin = kwargs['source']
        if origin[-4:] == ".mp4":
            VIDEO_FLAG = True
            print("**** DETECTED VIDEO FILE ****")
            shutil.copy(origin, path_2_data)
            self.source = path_2_source[:-6] + "swarm-data.mp4"
        else:
            VIDEO_FLAG = False
            print("**** DETECTED IMAGE FOLDER ****")
            src_files = os.listdir(origin)
            for file_name in src_files:
                full_file_name = os.path.join(origin, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, path_2_source)
                    self.source = path_2_source

        self.weights = kwargs['weights']
        self.scnn = kwargs['scnn']
        self.video = kwargs['video']
        self.debug = kwargs['debug']
        self.clean = kwargs['clean']

        environ = kwargs['environ']
        if environ:
            self.scnn = os.getenv("SCNN_SCNN")
            self.video = os.getenv("SCNN_VIDEO")
            self.debug = os.getenv("SCNN_DEBUG")
            self.clean = os.getenv("SCNN_CLEAN")
            self.weights = os.getenv("SCNN_WEIGHTS")

        self.base = "/".join(self.source.split("/")[:-1]) + "/"
        self.predict = self.base + "predicts/"
        self.destination = self.base + "Spliced"
        self.path_2_prob = self.base + "Prob"
        self.path_2_predict = self.predict + self.destination[5:] + "/"
        self.path_2_vid = self.base + "Videos"
        self.path_2_curves = self.base + "Curves"
        self.makedir(self.destination)


    '''
    def parse(self):
        # Optional method to check for commandline arguments and flags.
        parser = argparse.ArgumentParser()
        parser.add_argument("source", help="Path to video or image directory")
        parser.add_argument("-s", "--scnn", help="RUN SCNN probability map generation",
                            action="store_true", default=False)
        parser.add_argument("-v", "--video", help="RUN video generation",
                            action="store_true", default=False)
        parser.add_argument("-d", "--debug", help="RUN in debug mode (output displayed)",
                            action="store_true", default=False)
        args = parser.parse_args()
    '''

    @staticmethod
    def makedir(path):
        '''
        Check for existence of the folder first to prevent error generation.
        '''
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path)

    @staticmethod   
    @contextmanager
    def cd(new_dir):
        '''
        Extend context manager's "cd"; goes back to origin directory after completion.
        '''
        prev_dir = os.getcwd()
        os.chdir(os.path.expanduser(new_dir))
        try:
            yield
        finally:
            os.chdir(prev_dir)



    def splice(self):
        '''
        Use ffmpeg to generate key frames from video OR generate image directory (to generate lane
        curves, the coordinates text file and the respective frame needs to be in the same folder);
        therefore, a new directory is created for immutability of input. The directory is called
        "Spliced/".
        '''
        global VIDEO_FLAG

        if VIDEO_FLAG:
            print("**** SPLICING VIDEO INTO FRAMES ****")
            mp4_file = open(self.source, 'r')
            os.system("ffmpeg -loglevel panic -i {0} -r 0.1 {1}/%1d.jpg".format(mp4_file.name,
                                                                                self.destination))
            mp4_file.close()

        else:
            print("**** CREATING IMAGE DIRECTORY ****")
            for image in os.listdir(self.source):
                if image.endswith(".jpg"):
                    image_path = self.source + "/" + image
                    shutil.copy(image_path, self.destination)

            frames = os.listdir(self.destination)
            frames_sorted = sorted(([s.strip('.jpg') for s in frames]), key=int)
            count = 0

            for frame in frames_sorted:
                count = count + 1
                os.rename(self.destination + "/" + frame + ".jpg", self.destination + "/"
                          + str(count) + ".jpg")


    def make_test(self):
        '''
        SCNN requires a text file which points to every input frame. It uses that text file to
        generate probability maps. Make that txt file and place in correct directory.
        '''
        print("**** MAKING TEST.TXT FILE ****")
        global FRAMES_SORTED
        frames = os.listdir(self.destination)
        FRAMES_SORTED = sorted(([s.strip('.jpg') for s in frames]), key=int)
        txt = open(self.base + "test.txt", "w+")

        for i in range(len(FRAMES_SORTED)):
            txt.write("/" + self.destination[5:] + "/" + FRAMES_SORTED[i] + ".jpg" + "\n")

        txt.close()


    def resize(self):
        '''
        Resize image to comply with SCNN requirements.
        '''
        print("**** MAKING RESIZED IMAGES ****")
        global FRAMES_SORTED
        out_width = 1640
        out_height = 590

        for i in range(len(FRAMES_SORTED)):
            orig_image = Image.open(self.destination + "/" + str(i + 1) + ".jpg")
            scaled_image = orig_image.resize((out_width, out_height), Image.NEAREST)
            scaled_image.save(self.destination + "/" + str(i + 1) + ".jpg")


    def prob_maps(self):
        '''
        Simply, run the SCNN testing script to generate a probability map for each lane per frame
        (4 maps per frame). The probability maps are stored in the "predicts/" folder.
        '''
        if self.scnn:
            print("**** MAKING PROBABILITY MAPS ****")
            self.makedir(self.predict)

            model = "-model " + self.weights + " "
            data = "-data ./data "
            val = "-val " + self.base + "test.txt "
            save = "-save " + self.predict + " "
            dataset = "-dataset laneTest "
            share_grad_input = "-shareGradInput true "
            n_threads = "-nThreads 2 "
            n_gpu = "-nGPU 1 "
            batch_size = "-batchSize 1 "
            smooth = "-smooth true "

            if self.debug:
                os.system("rm ./gen/laneTest.t7")
                os.system("th testLane.lua " + model + data + val + save + dataset
                          + share_grad_input + n_threads + n_gpu + batch_size + smooth)
            else:
                os.system("rm ./gen/laneTest.t7 >/dev/null")
                os.system("th testLane.lua " + model + data + val + save + dataset \
                          + share_grad_input + n_threads + n_gpu + batch_size + smooth \
                          + ">/dev/null")

        else:
            print("**** BYPASSED: SCNN PROBABILITY MAPS ****")


    def avg_prob_maps(self):
        '''
        Add each lane probability map to generate per frame probability map (will all lanes in one
        image). Simple addition of probability maps is not ideal; therefore, this block might be
        removed in later releases (it is present for convenience). The results are stored in the
        "Prob/" folder.
        '''
        print("**** MAKING AVERAGES FROM PROBABILITY MAPS ****")
        self.makedir(self.path_2_prob)
        global FRAMES_SORTED

        for i in range(len(FRAMES_SORTED)):
            im_name = str(i + 1)
            im1_arr = numpy.asarray(Image.open(self.path_2_predict + im_name + "_1_avg.png"))
            im2_arr = numpy.asarray(Image.open(self.path_2_predict + im_name + "_2_avg.png"))
            im3_arr = numpy.asarray(Image.open(self.path_2_predict + im_name + "_3_avg.png"))
            im4_arr = numpy.asarray(Image.open(self.path_2_predict + im_name + "_4_avg.png"))
            #new_img = Image.blend(background, overlay, 0.5)
            #new_img = cv2.add(background, overlay)
            addition = im1_arr + im2_arr + im3_arr + im4_arr
            new_img = Image.fromarray(addition)
            new_img.save(self.path_2_prob + "/" + im_name + ".jpg", "JPEG")


    def lane_coord(self):
        '''
        Run matlab script to make lane coordinates. The coordinates are stored in
        [frame no.].lines.txt" in the "Spliced/" folder.
        '''
        print("**** MAKING LANE COORDINATES ****")

        with self.cd("./tools/prob2lines"):
            path_2_scnn = "../../"
            exp1 = self.weights
            data1 = path_2_scnn + "data"
            prob_root1 = path_2_scnn + self.predict + self.destination[5:]
            output1 = path_2_scnn + self.destination
            mat_args = "\'" + exp1 + "\', \'" + data1 + "\', \'" \
                    + prob_root1 + "\', \'" + output1 + "\'"

            if self.debug:
                os.system("matlab -nodisplay -r \"try coords(" + mat_args \
                          + "); catch; end; quit\"")
            else:
                os.system("matlab -nodisplay -r \"try coords(" + mat_args \
                          + "); catch; end; quit\" >/dev/null")


    def lane_curve(self):
        '''
        Run seg_label_generate to create lane curves by using a cubic spline. The curves are on
        top of the input image and stored in a separate folder called "Curves/".
        flag details are:
          -l: image list file to process
          -m: set mode to "imgLabel" or "trainList"
          -d: dataset path
          -w: the width of lane labels generated
          -o: path to save the generated labels
          -s: visualize annotation, remove this option to generate labels
        '''
        print("**** MAKING LANE CURVES ****")
        self.makedir(self.path_2_curves)

        path_2_scnn = "../"
        list_file = "-l " + path_2_scnn + self.base + "test.txt "
        mode = "-m " + "imgLabel "
        data = "-d " + path_2_scnn + "data "
        width = "-w " + "16 "
        output = "-o " + path_2_scnn + "data "
        vis = "-s "

        with self.cd("./seg_label_generate"):
            os.system("make clean")
            if self.debug:
                os.system("make")
                os.system("./seg_label_generate " + list_file + mode + data + width + output + vis)
            else:
                os.system("make >/dev/null")
                os.system("./seg_label_generate " + list_file + mode + data + width + output \
                          + vis + ">/dev/null")


    def gen_video(self):
        '''
        Use ffmpeg to generate videos using given key frames with lane curves
        '''
        if self.video:
            print("**** MAKING ALL VIDEOS ****")
            self.makedir(self.path_2_vid)
            os.system("ffmpeg -loglevel panic -framerate 24 -i {0}/%1d.jpg {1}/prob.mp4".format(self.path_2_prob, self.path_2_vid))
            os.system("ffmpeg -loglevel panic -framerate 24 -i {0}/%1d.jpg {1}/curve.mp4".format(self.path_2_curves, self.path_2_vid))
        else:
            print("**** BYPASSED: VIDEO GENERATION ****")


    def check_lanes(self):
        '''
        Read "[frame no.].exist.txt" to check how many lanes there are
        '''
        print("**** FINDING NUMBER OF LANES ****")
        global FRAMES_SORTED
        global FRAME_LANES

        for i in range(len(FRAMES_SORTED)):
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
            FRAME_LANES[int(frame)] = final_list
            exist_txt.close()


    def check_current_lane(self):
        '''
        To find out which lane the car is in:
        1. Read "[frame no.].lines.txt generated by the matlab code to obtain all coordinates for
        each frame
        2. Check whether a coordinate exists (for each detected lane) that is on the bottom-left
        quadrant
        3. If it does, then a lane exists that is on the left side of the car
        Note: a key assumption is that the camera is always on the middle of the car i.e. for each
        frame, the car's location is in the middle. This is true for 99% of the cases; however, if
        not, then offset xCoord by whatever value necessary.
        '''
        print("**** FINDING CURRENT LANE ****")
        global FRAMES_SORTED
        global FRAME_LANES

        for i in range(len(FRAMES_SORTED)):
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

                    if found_lane is False:
                        if (x_coord < 700) and (y_coord > 350):
                            found_lane = True
                            count = count + 1

            FRAME_LANES[int(frame)].append(count)
            coord_txt.close()


    def conf(self):
        '''
        Read "[frame no.].conf.txt" to check what the confidence for each lane detected is.
        '''
        print("**** FINDING CONFIDENCE ****")
        global FRAMES_SORTED
        global FRAME_LANES

        for i in range(len(FRAMES_SORTED)):
            frame = str(i + 1)
            conf_name = self.path_2_predict + frame + ".conf.txt"
            conf_txt = open(conf_name, "r")
            conf_read = conf_txt.read().replace("\n", "")
            conf_list = conf_read.split(" ")[:-1]
            conf_list = [float(i) for i in conf_list]
            FRAME_LANES[int(frame)].append(conf_list)
            conf_txt.close()


    def json(self):
        '''
        Generate my bruh jason
        '''
        print("**** MAKING JSON OUTPUT ****")
        json_file = open(self.base + "data.json", "w+")
        data = []
        global FRAMES_SORTED
        global FRAME_LANES

        for i in range(len(FRAMES_SORTED)):
            frame = i + 1
            count = FRAME_LANES[frame][0]
            lane = FRAME_LANES[frame][1]
            conf = FRAME_LANES[frame][2]
            data.append({"frame": frame, "lanes_count": count,
                         "current_lane": lane, "confidence": conf})

        json.dump(data, json_file, indent=4)
        json_file.close()

        return json.dumps(data, indent=4)


    def clean_all(self):
        '''
        Clean all temporary files if relevant flag passes.
        '''
        if self.clean:
            print("**** CLEANING TEMPORARY FILES AND FOLDERS ****")
            shutil.rmtree(self.path_2_curves)
            os.remove(self.base + "data.json")
            shutil.rmtree(self.predict)
            shutil.rmtree(self.path_2_prob)
            if VIDEO_FLAG:
                os.remove(self.source)
            else:
                shutil.rmtree(self.source)
            shutil.rmtree(self.destination)
            os.remove(self.base + "test.txt")


    def run_all(self):
        '''
        Run the whole pipeline
        '''
        time_1 = time.time()

        self.splice()
        self.make_test()
        self.resize()

        time_scnn_1 = time.time()
        self.prob_maps()
        time_scnn = time.time() - time_scnn_1

        self.avg_prob_maps()
        time_matlab_1 = time.time()
        self.lane_coord()
        time_matlab = time.time() - time_matlab_1

        time_seg_1 = time.time()
        self.lane_curve()
        time_seg = time.time() - time_seg_1

        self.gen_video()

        time_post_1 = time.time()
        self.check_lanes()
        self.check_current_lane()
        self.conf()
        ret = self.json()
        self.clean_all()
        time_post = time.time() - time_post_1

        time_all = time.time() - time_1
        num = len(FRAMES_SORTED)
        fps = num / time_all
        print("**** TIME TO RUN SCNN: {0}".format(time_scnn))
        print("**** TIME TO GET COORDS: {0}".format(time_matlab))
        print("**** TIME TO MAKE CURVES: {0}".format(time_seg))
        print("**** TIME TO GET INFO: {0}".format(time_post))
        print("**** TOTAL TIME: {0}".format(time_all))
        print("**** FRAMES PER SECOND: {0}".format(fps))

        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Path to video or image directory")
    parser.add_argument("-w", "--weights", help="Path to weights file",
                        default='experiments/pretrained/model_best_rz.t7')
    parser.add_argument("-e", "--environ", help="USE Environment Variables instead",
                        action="store_true", default=False)
    parser.add_argument("-s", "--scnn", help="RUN SCNN probability map generation",
                        action="store_true", default=False)
    parser.add_argument("-v", "--video", help="RUN video generation",
                        action="store_true", default=False)
    parser.add_argument("-d", "--debug", help="RUN in debug mode (output displayed)",
                        action="store_true", default=False)
    parser.add_argument("-c", "--clean", help="REMOVE all generated folders and files",
                        action="store_true", default=False)
    kwargs = vars(parser.parse_args())

    scnn_test = SCNN(**kwargs)
    scnn_test.run_all()
