#!/bin/bash

# modify CULane to yours
CULane=~/SCNN/data
OutputPath=${CULane}
if [ ! -d $OutputPath ]; then
  mkdir $OutputPath
fi
./seg_label_generate \
    -l ${CULane}/example-swarm-data/test.txt \
    -m imgLabel \
    -d $CULane \
    -w 16 \
    -o $OutputPath \
    -s
# explanation:
# -l: image list file to process
# -m: set mode to "imgLabel" or "trainList"
# -d: dataset path
# -w: the width of lane labels generated
# -o: path to save the generated labels
# -s: visualize annotation, remove this option to generate labels
