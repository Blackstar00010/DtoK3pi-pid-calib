#!/bin/bash

FILENAME=../../../logs/$(date +'%Y%m%d/%H%M%S')_root.log
mkdir -p $(dirname $FILENAME)
touch $FILENAME

# running roc.py doesn't do anything but printing values; it is used to generate roc.csv using `python roc.py > roc.csv`
# python roc.py 2>&1 | tee -a $FILENAME
python roc.py > roc.csv 2>&1 | tee -a $FILENAME

root -q fitplots.cpp 2>&1 | tee -a $FILENAME
root -q plots.cpp 2>&1 | tee -a $FILENAME
root -q plotCompRep.cpp 2>&1 | tee -a $FILENAME

cp $FILENAME ../../../logs/most_recent.log

echo Log file saved to $FILENAME
echo Log file also saved as ../../../logs/most_recent.log
