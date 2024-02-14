#include <iostream>
#include <sstream>
#include <cublasLt.h>
#include <stdexcept>

inline void cuda_check_error(const cudaError_t error, const std::string filename, const std::size_t line, const std::string funcname, const std::string message = ""){
  if(error != cudaSuccess){
    std::stringstream ss;
    ss << cudaGetErrorString( error );
    if(message.length() != 0){
      ss << " : " << message;
    }
    ss << " [" << filename << ":" << line << " in " << funcname << "]";
    throw std::runtime_error(ss.str());
  }
}

inline void cuda_check_error(const cublasStatus_t error, const std::string filename, const std::size_t line, const std::string funcname, const std::string message = ""){
  if(error != CUBLAS_STATUS_SUCCESS){
    std::stringstream ss;
    ss << cublasGetStatusString(error);
    if(message.length() != 0){
      ss << " : " << message;
    }
    ss << " [" << filename << ":" << line << " in " << funcname << "]";
    throw std::runtime_error(ss.str());
  }
}
#ifndef CUDA_CHECK_ERROR
#define CUDA_CHECK_ERROR(status) cuda_check_error(status, __FILE__, __LINE__, __func__)
#endif


template <class T>
void eval(const std::size_t N) {
  T *dev_a, *dev_b, *dev_c;
  CUDA_CHECK_ERROR(cudaMallocManaged(&dev_a, sizeof(T) * N * N));
  CUDA_CHECK_ERROR(cudaMallocManaged(&dev_b, sizeof(T) * N * N));
  CUDA_CHECK_ERROR(cudaMallocManaged(&dev_c, sizeof(T) * N * N));

#pragma omp parallel for
  for (std::size_t i = 0; i < N * N; i++) {
    dev_a[i] = i;
    dev_b[i] = i + 3;
    dev_c[i] = i + 2;
  }

  CUDA_CHECK_ERROR(cudaMemAdvise(dev_a, sizeof(T) * N * N, cudaMemAdviseSetAccessedBy, 0));
  CUDA_CHECK_ERROR(cudaMemAdvise(dev_b, sizeof(T) * N * N, cudaMemAdviseSetAccessedBy, 0));
  CUDA_CHECK_ERROR(cudaMemAdvise(dev_c, sizeof(T) * N * N, cudaMemAdviseSetAccessedBy, 0));

  cublasLtHandle_t handle;
  CUDA_CHECK_ERROR(cublasLtCreate(&handle));

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t desc_a = nullptr, desc_b = nullptr, desc_c = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  const cublasOperation_t trans_a = CUBLAS_OP_N, trans_b = CUBLAS_OP_N;

  CUDA_CHECK_ERROR(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  CUDA_CHECK_ERROR(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
  CUDA_CHECK_ERROR(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

  CUDA_CHECK_ERROR(cublasLtMatrixLayoutCreate(&desc_a, CUDA_R_32F, N, N, N));
  CUDA_CHECK_ERROR(cublasLtMatrixLayoutCreate(&desc_b, CUDA_R_32F, N, N, N));
  CUDA_CHECK_ERROR(cublasLtMatrixLayoutCreate(&desc_c, CUDA_R_32F, N, N, N));

  CUDA_CHECK_ERROR(cublasLtMatmulPreferenceCreate(&preference));
  std::size_t workspace_size = 4lu << 20;
  CUDA_CHECK_ERROR(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
  void* workspace;
  CUDA_CHECK_ERROR(cudaMalloc(&workspace, workspace_size));

  int returned_results = 0;
  cublasLtMatmulHeuristicResult_t heuristic_result = {};
  CUDA_CHECK_ERROR(cublasLtMatmulAlgoGetHeuristic(handle, op_desc, desc_a, desc_b, desc_c, desc_c, preference, 1, &heuristic_result, &returned_results));

  const T alpha = 1, beta = 0;
  CUDA_CHECK_ERROR(cublasLtMatmul(
      handle,
      op_desc,
      &alpha,
      dev_a, desc_a,
      dev_b, desc_b,
      &beta,
      dev_c, desc_c,
      dev_c, desc_c,
      &heuristic_result.algo,
      workspace,
      workspace_size,
      0
      ));
  CUDA_CHECK_ERROR(cudaDeviceSynchronize());

  CUDA_CHECK_ERROR(cublasLtMatmulPreferenceDestroy(preference));
  CUDA_CHECK_ERROR(cublasLtMatrixLayoutDestroy(desc_a));
  CUDA_CHECK_ERROR(cublasLtMatrixLayoutDestroy(desc_b));
  CUDA_CHECK_ERROR(cublasLtMatrixLayoutDestroy(desc_c));
  CUDA_CHECK_ERROR(cublasLtMatmulDescDestroy(op_desc));

  CUDA_CHECK_ERROR(cublasLtDestroy(handle));

  CUDA_CHECK_ERROR(cudaFree(workspace));
  CUDA_CHECK_ERROR(cudaFree(dev_a));
  CUDA_CHECK_ERROR(cudaFree(dev_b));
  CUDA_CHECK_ERROR(cudaFree(dev_c));
}

int main() {
  eval<float>(1lu << 14);
}
