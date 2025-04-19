# Copyright (c) 2025 Kulagin Aleksandr

LAB_NAME = gemm_block_oneapi

CXX = icpx

CXXFLAGS = -Wall -g -O3 -std=c++17 -pedantic -fsycl

TARGET = $(LAB_NAME).bin

SRCS = main.cpp $(LAB_NAME).cpp

OBJS = $(SRCS:.cpp=.o)

all: $(TARGET) run clean

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)
