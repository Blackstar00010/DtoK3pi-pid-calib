#!/bin/bash

FILENAME=../../../logs/$(date +'%Y%m%d/%H%M%S')_root.log
mkdir -p $(dirname $FILENAME)
touch $FILENAME

root -q plotWeighted.cpp 2>&1 | tee -a $FILENAME
root -q plotEff.cpp 2>&1 | tee -a $FILENAME
root -q plotROC.cpp 2>&1 | tee -a $FILENAME
root -q plotComp.cpp 2>&1 | tee -a $FILENAME

cp $FILENAME ../../../logs/most_recent.log

echo Log file saved to $FILENAME
echo Log file also saved as ../../../logs/most_recent.log
