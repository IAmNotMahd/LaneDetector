#!/bin/bash
# Installing Torch
if echo $* | grep -e "-t" -q; then
	echo "installing torch"
	git clone https://github.com/torch/distro.git ~/torch --recursive
	cd ~/torch; bash install-deps;
	cd ~/torch; bash install-deps;
	./install.sh
	source ~/.bashrc
fi
# Downloading Dataset
if echo $* | grep -e "-d" -q; then
	echo "Downloading Dataset"
	mkdir -p SCNN/data/CULane
	cd SCNN/data/CULane
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QbB1TOk9Fy6Sk0CoOsR3V8R56_eG6Xnu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1QbB1TOk9Fy6Sk0CoOsR3V8R56_eG6Xnu" -O annotations_new.tar.gz && rm -rf /tmp/cookies.txt
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=14Gi1AXbgkqvSysuoLyq1CsjFSypvoLVL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=14Gi1AXbgkqvSysuoLyq1CsjFSypvoLVL" -O driver_23_30frame.tar.gz && rm -rf /tmp/cookies.txt
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Z6a463FQ3pfP54HMwF3QS5h9p2Ch3An7' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Z6a463FQ3pfP54HMwF3QS5h9p2Ch3An7" -O driver_37_30frame.tar.gz && rm -rf /tmp/cookies.txt
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1LTdUXzUWcnHuEEAiMoG42oAGuJggPQs8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LTdUXzUWcnHuEEAiMoG42oAGuJggPQs8" -O driver_100_30frame.tar.gz && rm -rf /tmp/cookies.txt
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1AQjQZwOAkeBTSG_1I9fYn8KBcxBBbYyk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1AQjQZwOAkeBTSG_1I9fYn8KBcxBBbYyk" -O driver_161_90frame.tar.gz && rm -rf /tmp/cookies.txt
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1PH7UdmtZOK3Qi3SBqtYOkWSH2dpbfmkL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1PH7UdmtZOK3Qi3SBqtYOkWSH2dpbfmkL" -O driver_182_30frame.tar.gz && rm -rf /tmp/cookies.txt
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1daWl7XVzH06GwcZtF4WD8Xpvci5SZiUV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1daWl7XVzH06GwcZtF4WD8Xpvci5SZiUV" -O driver_193_90frame.tar.gz && rm -rf /tmp/cookies.txt
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yCOXaaNcyoVrHDR0_A_gXH-thg-7QDv8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yCOXaaNcyoVrHDR0_A_gXH-thg-7QDv8" -O laneseg_label_w16_test.zip && rm -rf /tmp/cookies.txt
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1MlL1oSiRu6ZRU-62E39OZ7izljagPycH' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1MlL1oSiRu6ZRU-62E39OZ7izljagPycH" -O laneseg_label_w6.tar.gz && rm -rf /tmp/cookies.txt
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=18alVEPAMBA9Hpr3RDAAchqSj5IxZNRKd' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18alVEPAMBA9Hpr3RDAAchqSj5IxZNRKd" -O list.tar.gz && rm -rf /tmp/cookies.txt

	tar -xzf annotations_new.tar.gz
	tar -xzf driver_23_30frame.tar.gz
	tar -xzf driver_37_30frame.tar.gz 
	tar -xzf driver_100_30frame.tar.gz
	tar -xzf driver_161_90frame.tar.gz
	tar -xzf driver_182_30frame.tar.gz
	tar -xzf driver_193_90frame.tar.gz
	tar -xzf laneseg_label_w16.tar.gz
	tar -xzf list.tar.gz

	rm annotations_new.tar.gz
	rm driver_23_30frame.tar.gz
	rm driver_37_30frame.tar.gz
	rm driver_100_30frame.tar.gz
	rm driver_161_90frame.tar.gz
	rm driver_182_30frame.tar.gz
	rm driver_193_90frame.tar.gz
	rm laneseg_label_w16.tar.gz
	rm list.tar.gz
	cd ..
	cd ..
	cd ..
fi
# Downloading weights file
if echo $* | grep -e "-w" -q; then
	echo "Downloading Weights"
	cd SCNN/experiments/pretrained
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Wv3r3dCYNBwJdKl_WPEfrEOt-XGaROKu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Wv3r3dCYNBwJdKl_WPEfrEOt-XGaROKu" -O vgg_SCNN_DULR_w9.t7 && rm -rf /tmp/cookies.txt
	cd ..
	cd ..
	cd ..
fi
# Replacing files with custom files
cp lane_detector/frameList.py SCNN/data/
cp lane_detector/coords.m SCNN/tools/prob2lines/
cp lane_detector/seg_label_generator.cpp SCNN/seg_label_generate/src/
cp lane_detector/seg_label_generator.hpp SCNN/seg_label_generate/include/
cp lane_detector/testLane.lua SCNN/
