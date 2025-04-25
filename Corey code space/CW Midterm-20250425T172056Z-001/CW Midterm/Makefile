# Compiler
CC = g++

# Include OpenCV headers explicitly
CFLAGS = -Wall -std=c++11 -I/usr/include/opencv4

# Link OpenCV and other required libraries
LDLIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui \
         -lopencv_imgcodecs -lopencv_calib3d \
         -ljpeg -lm -llapacke -llapack -lblas

# Source files
SRCS = stereo.cc pRectify.cc imageUtils.cc
OBJS = $(SRCS:.cc=.o)

# Output executables
P_RECTIFY_BIN = pRectify
STEREO_BIN = stereo

# Default target builds both
all: $(P_RECTIFY_BIN) $(STEREO_BIN)

$(P_RECTIFY_BIN): pRectify.o imageUtils.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(STEREO_BIN): stereo.o imageUtils.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cc
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(P_RECTIFY_BIN) $(STEREO_BIN)

run: all
	./pRectify
	./stereo
