all: main

ppm: main
	./main > test.ppm

main: Main.cu
	nvcc Main.cu -o main
