# Copyright (c) 2025 Kulagin Aleksandr
CXX = g++

CXXFLAGS = -Wall -g -O3 -std=c++11 -pedantic

TARGET = permutations_cxx_test.bin

SRCS = main.cpp permutations_cxx.cpp

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
