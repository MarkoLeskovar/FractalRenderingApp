#!/bin/bash

# Configure cmake files
printf "\nCONFIGURING RELEASE BUILD...\n"
cmake -S source/ -B build/ -DPYBIND11_FINDPYTHON=ON -DCMAKE_BUILD_TYPE=Release

# Build the files
printf "\nBUILDING...\n"
cmake --build build/
printf "\n"
