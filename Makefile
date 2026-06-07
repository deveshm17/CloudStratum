# Cloud Job Scheduler — Makefile

CXX      = g++
CXXFLAGS = -std=c++17 -O2
BIN      = optimizer/optimizer
SRC      = optimizer/optimizer.cpp

.PHONY: all install build run run-fast clean

## Install Python dependencies
install:
	pip install -r requirements.txt

## Compile C++ optimizer
build:
	$(CXX) $(CXXFLAGS) -o $(BIN) $(SRC)
	@echo "C++ optimizer compiled → $(BIN)"

## Run full pipeline (data → ML → C++ optimizer → ILP → evaluate)
run: build
	python main.py

## Run without ILP (faster, skips OR-Tools)
run-fast: build
	python main.py --skip-ilp --sa-iter 30000

## Run with more SA iterations for better quality
run-quality: build
	python main.py --sa-iter 100000 --sa-temp 1000.0

## Clean compiled binary and output
clean:
	rm -f $(BIN)
	rm -f output/*.json

## Show project structure
tree:
	find . -not -path '*/.*' -not -path '*/ml/models/*' -not -path '*/__pycache__/*' | sort | sed 's|[^/]*/|  |g'
