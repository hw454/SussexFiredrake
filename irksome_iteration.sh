#!/bin/bash
# Keith Briggs 2020-11-02 & Hayley Wragg
# bash Raytracer_transmittermover.sh
for ((job=0; job<=109; job++)); do
    echo "job=${job}"
    time python3 irksome_fullfunc.py "${job}"
done

