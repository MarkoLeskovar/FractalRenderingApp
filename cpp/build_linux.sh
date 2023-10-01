#!/bin/bash


# Configure cmake files
printf "\nCONFIGURING...\n"
cmake -S source/ -B build/

# Build the files
printf "\nBUILDING...\n"
cd build
make
printf "\n"
