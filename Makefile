default:
	nvcc -g match.cu -o match

run: default
	./match > output

clean:
	rm -f match