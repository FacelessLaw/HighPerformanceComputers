GPU_COMPILER=nvcc
GPU_FLAGS=-O3 -std=c++11 -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm

main-gpu: main_cuda.cu
	$(GPU_COMPILER) $(GPU_FLAGS) main_cuda.cu -o main

submit-polus-gpu: main-gpu
	bsub -n 4 -W 00:05 -gpu "num=1" -R "span[ptile=2]" -o stdout.txt -e error.txt OMP_NUM_THREADS=1 mpiexec ./main

submit-polus-nvprof: main-gpu
	bsub -n 1 -W 00:05 -gpu "num=1" -R "span[ptile=2]" -o stdout.txt -e error.txt OMP_NUM_THREADS=1 mpiexec nvprof ./main
