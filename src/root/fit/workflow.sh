#!/bin/bash

FILENAME=../../../logs/$(date +'%Y%m%d/%H%M%S')_root.log
mkdir -p $(dirname $FILENAME)
touch $FILENAME

# # to simply run files, use:
# python jointworoot.py 2>&1 | tee -a $FILENAME # this takes the longest
root -q fit2d.cpp 2>&1 | tee -a $FILENAME
python sortpis.py 2>&1 | tee -a $FILENAME

cp $FILENAME ../../../logs/most_recent.log

echo Log file saved to $FILENAME
echo Log file also saved as ../../../logs/most_recent.log

# # to debug files, use:
# export CLING_DEBUG=1
# g++ -g -O0 -Wall shit.cpp $(root-config --cflags --libs) -o shit
# gdb ./shit
# # then, in the gdb console, use `bt full``
