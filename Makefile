def2:
	nvcc -g transform.cu im1.cc -I. -I/usr/include/OpenEXR -lIlmImf -lImath -lHalf -o trans -arch=sm_21
r2: def2
	optirun ./trans stillife.exr

default:
	nvcc -g match.cu -o match

run: default
	./match > output

clean:
	rm -f match trans