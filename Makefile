CXX = nvcc

OUTPUT = image_stab

CXXSOURCES = 	mainSift.cpp \
							geomFuncs.cpp \
							im1.cpp

CUDASOURCES =	cudaImage.cu \
							cudaSiftH.cu \
							error_handling.cu \
							match.cu \
							match_trans.cu \
							matching.cu \
							sift.cu \
							singular.cu \
							transform.cu

HFILES = $(wildcard *.h)

IDIR = -I. -I/usr/include/OpenEXR -I/usr/local/include/opencv -I/usr/local/include/opencv2

OBJS = $(CXXSOURCES:.cpp=.o) $(CUDASOURCES:.cu=.o)

CXXFLAGS = -g

NVCCFLAGS = -arch=sm_21 -g

LFLAGS = -lIlmImf -lImath -lHalf -lopencv_core -lopencv_imgproc -lopencv_highgui

$(OUTPUT) : $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LFLAGS)

%.o: %.cc
	$(CXX) $(NVCCFLAGS) $(IDIR) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(NVCCFLAGS) $(IDIR) -c -o $@ $<

%.o: %.cu $(HFILES)
	$(CXX) $(NVCCFLAGS) $(IDIR) -c -o $@ $<

clean:
	rm -f $(OBJS) $(OUTPUT)
