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

  cublasLtHandle_t handle;
  cublasLtCreate(&handle);

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t desc_a = nullptr, desc_b = nullptr, desc_c = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  cublasOperation_t trans_a, trans_b;

  cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
  cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));

  cublasLtMatrixLayoutCreate(&desc_a, CUDA_R_32F, N, N, N);
  cublasLtMatrixLayoutCreate(&desc_b, CUDA_R_32F, N, N, N);
  cublasLtMatrixLayoutCreate(&desc_c, CUDA_R_32F, N, N, N);

  cublasLtMatmulPreferenceCreate(&preference);
  std::size_t workspace_size;
  cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size));
  void* workspace;
  cudaMalloc(&workspace, 4lu << 20);

  int returned_results = 0;
  cublasLtMatmulHeuristicResult_t heuristic_result = {};
  cublasLtMatmulAlgoGetHeuristic(handle, op_desc, desc_a, desc_b, desc_c, desc_c, preference, 1, &heuristic_result, &returned_results);

  const T alpha = 1, beta = 0;
  cublasLtMatmul(
      handle,
      op_desc,
      &alpha,
      dev_a, desc_a,
      dev_b, desc_b,
      &beta,
      desc_c, desc_c,
      desc_c, desc_c,
      &heuristic_result.algo,
      workspace,
      workspace_size,
      0
      );

  cublasLtMatmulPreferenceDestroy(preference);
  cublasLtMatrixLayoutDestroy(desc_a);
  cublasLtMatrixLayoutDestroy(desc_b);
  cublasLtMatrixLayoutDestroy(desc_c);
  cublasLtMatmulDescDestroy(op_desc);

  cublasLtDestroy(handle);

  cudaFree(workspace);
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
}

int main() {
  eval<float>(1lu << 14);
}
