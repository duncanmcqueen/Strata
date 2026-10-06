#pragma once
// include/strata/sycl_compat/cublas_v2.h - the cuBLAS subset Strata uses (gemm.cu),
// declared here and implemented by src/sycl_runtime/blas.cpp on oneMKL's SYCL BLAS.
// cuBLAS is column-major; oneMKL's column_major::gemm maps 1:1, with the queue coming
// from cublasSetStream.  Workspace and math mode are accepted and ignored (oneMKL
// manages both).

#include <cstddef>

enum cublasStatus_t {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
};

enum cublasOperation_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 };

// The cudaDataType values cuBLAS call sites pass (library_types.h in a CUDA build).
enum cudaDataType_t {
    CUDA_R_32F = 0,
    CUDA_R_64F = 1,
    CUDA_R_16F = 2,
    CUDA_R_8I = 3,
    CUDA_R_16BF = 14,
};
typedef cudaDataType_t cudaDataType;

enum cublasComputeType_t { CUBLAS_COMPUTE_32F = 68, CUBLAS_COMPUTE_32F_FAST_TF32 = 77 };
enum cublasMath_t { CUBLAS_DEFAULT_MATH = 0, CUBLAS_TF32_TENSOR_OP_MATH = 3 };
enum cublasGemmAlgo_t { CUBLAS_GEMM_DEFAULT = -1 };

typedef struct strata_cublas_handle* cublasHandle_t;

cublasStatus_t cublasCreate(cublasHandle_t* handle);
cublasStatus_t cublasDestroy(cublasHandle_t handle);
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream);
cublasStatus_t cublasSetWorkspace(cublasHandle_t handle, void* workspace, size_t workspaceBytes);
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);
cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                            int k, const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B,
                            cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc,
                            cublasComputeType_t computeType, cublasGemmAlgo_t algo);
