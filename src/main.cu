#include <iostream>
#include <cublasLt.h>

template <class T>
void eval(const std::size_t N) {
  T *dev_a, *dev_b, *dev_c;
  cudaMallocManaged(&dev_a, sizeof(T) * N * N);
  cudaMallocManaged(&dev_b, sizeof(T) * N * N);
  cudaMallocManaged(&dev_c, sizeof(T) * N * N);

#pragma omp parallel for
  for (std::size_t i = 0; i < N * N; i++) {
    dev_a[i] = i;
    dev_b[i] = i + 3;
    dev_c[i] = i + 2;
  }
}
