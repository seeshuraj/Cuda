NVCC = /usr/local/cuda-12.8/bin/nvcc
CXX = g++
CXXFLAGS = -O3 -std=c++11 -I/usr/local/cuda-12.8/include

all: ei_exec

ei_exec: main.o cpu_integral.o gpu_integral.o gpu_launcher.o
	$(NVCC) -o ei_exec main.o cpu_integral.o gpu_integral.o gpu_launcher.o -lcuda -lcudart

main.o: main.cpp gpu_integral.h cpu_integral.h
	$(CXX) $(CXXFLAGS) -c main.cpp

cpu_integral.o: cpu_integral.cpp cpu_integral.h
	$(CXX) $(CXXFLAGS) -c cpu_integral.cpp

gpu_integral.o: gpu_integral.cu
	$(NVCC) -O3 --expt-relaxed-constexpr -c gpu_integral.cu

gpu_launcher.o: gpu_launcher.cu gpu_integral.h
	$(NVCC) -O3 --expt-relaxed-constexpr -c gpu_launcher.cu

clean:
	rm -f *.o ei_exec
