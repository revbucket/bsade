#!/bin/bash 
# A setup script to install all packages and get ready to run bsade jobs
# Designed to be as lightweight as possible and just have the installs for bsade only
# (i.e., any tooling for setting up the file system or downloading data is _not_ included here!)

pip install pybind11 

# Try the first compilation command
echo "Starting compilation..."
if c++ -std=c++20 -O3 -shared -fPIC $(python3 -m pybind11 --includes) cpp_engine_dedup.cpp -o cpp_engine_dedup$(python3-config --extension-suffix); then
    echo "Compiled successfully!"
else
    echo "First compilation failed, trying with -undefined dynamic_lookup flag..."
    if c++ -std=c++20 -O3 -shared -fPIC $(python3 -m pybind11 --includes) cpp_engine_dedup.cpp -o cpp_engine_dedup$(python3-config --extension-suffix) -undefined dynamic_lookup; then
        echo "Compiled successfully with fallback method!"
    else
        echo "ERROR: Both compilation attempts failed!"
        exit 1
    fi
fi