# DtoK3pi-pid-calib

PID calibration of RICH detectors using the D➝K3pi decay channel

## Description

This repo contains the code used to study the PID performance of the RICH detector of the LHCb experiment using the D➝K3pi decay channel. For the research, python(pandas+uproot) was used to train BDT and filter out most of the backgrounds, and ROOT using C++ was implemented to find fits, calculate sWeights, and plot the results.

The repo does not contain any of the data and the results, as they are not publicly available. This repo is for educational purposes only, for those who seek some sample code to start with.

A lot of the code that are not the part of the final workflow have not been maintained for quite some time, therefore they are highly likely to be broken.
