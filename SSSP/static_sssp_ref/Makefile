# -*- MakeFile -*-

CC = g++
NVCC = nvcc

all: sssp

sssp: obj/main.o obj/dijkstra.o obj/utils.o obj/sssp.o obj/sssp_kernel.o
	$(NVCC) obj/main.o obj/dijkstra.o obj/utils.o obj/sssp.o obj/sssp_kernel.o -o sssp

obj/main.o: main.cu obj/dijkstra.o obj/utils.o obj/sssp.o obj/sssp_kernel.o obj
	$(NVCC) -c main.cu -o obj/main.o

obj/dijkstra.o: src/dijkstra.cpp include/graph.h obj
	$(CC) -c src/dijkstra.cpp -o obj/dijkstra.o

obj/utils.o: src/utils.cpp include/graph.h obj
	$(CC) -c src/utils.cpp -o obj/utils.o

obj/sssp.o: src/sssp.cu include/graph.cuh include/graph.h obj
	$(NVCC) -c src/sssp.cu -o obj/sssp.o

obj/sssp_kernel.o: src/sssp_kernel.cu include/graph.cuh obj
	$(NVCC) -c src/sssp_kernel.cu -o obj/sssp_kernel.o

obj:
	mkdir obj

clean:
	rm -f obj/*.o ./sssp
	rmdir obj