# File       : Makefile
# Created    : Tue Oct 16 2018 10:53:24 AM (+0200)
# Description: Compile targets
# Copyright 2018 ETH Zurich. All Rights Reserved.
CXX ?= g++
###############################################################################
CXXFLAGS = -O2 -msse -march=native -std=c++11 -fopenmp -Wall -Wextra
###############################################################################

all: power_method

power_method: power_method.cpp Makefile
	$(CXX) $(CXXFLAGS) power_method.cpp -lblas -o power_method

run:
	./power_method

plot:
	./plot

runplot: run plot

clean:
	rm -f power_method
	rm -rf MyEigenvalue.txt MyOwnTime.txt MyCblasTime.txt MyIter.txt
	rm -rf results.png

.PHONY: all clean run plot runplot
