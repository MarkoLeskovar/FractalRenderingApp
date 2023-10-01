#!/bin/bash

# Configure cmake files
printf "\nCONFIGURING RELEASE BUILD...\n"
cmake -S source/ -B build/ -DPYBIND11_FINDPYTHON=ON

# Build the files
printf "\nBUILDING...\n"
cmake --build build/ --config Release
printf "\n"
