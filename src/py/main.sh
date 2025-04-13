#!/bin/zsh

# prepare data
source prep/main.sh

# classify and save scores
source classify/main.sh

# fit and find sweights
source fit/main.sh

# plot
source plot/main.sh