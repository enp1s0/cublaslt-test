NVCC=nvcc
NVCCFLAGS=-std=c++17 -Xcompiler="-Wall -fopenmp"
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80
NVCCFLAGS+=-lcublasLt -lcublas

TARGET=cublaslt.test

$(TARGET):src/main.cu
	$(NVCC) $< -o $@ $(NVCCFLAGS)
  
clean:
	rm -f $(TARGET)
