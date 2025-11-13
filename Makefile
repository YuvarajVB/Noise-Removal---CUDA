NVCC = nvcc
CFLAGS = -std=c++17 -O2
OPENCV = `pkg-config --cflags --libs opencv4`
TARGET = bin/noise_removal

SRC = src/main.cu src/noise_removal.cu

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p bin
	$(NVCC) $(CFLAGS) $(SRC) -o $(TARGET) $(OPENCV)

clean:
	rm -rf bin/*
