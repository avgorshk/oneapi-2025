# Copyright (c) 2025 Kulagin Aleksandr

CXX = icpx

CXXFLAGS = -Wall -g -O3 -std=c++17 -pedantic -fsycl

TARGET = integral_oneapi.bin

SRCS = main.cpp integral_oneapi.cpp

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
