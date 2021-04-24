#!/bin/bash

#curl "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download" --output ./WESAD.zip

unzip ./WESAD -d ./

python data_wrangling.py
