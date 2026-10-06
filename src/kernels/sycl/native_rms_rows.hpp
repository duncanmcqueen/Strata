#pragma once
// src/kernels/sycl/native_rms_rows.hpp - private to the SYCL kernels: native_gr_norm.cpp's weighted RMSNorm with
// the gamma matrix repeating every `gamma_rows` rows.  native_gr_rms_norm_weighted is this with gamma_rows ==
// n_rows; native PLE's token batch passes the stream count, so its rows run the same kernel as single tokens'.
namespace strata::kernels::sycl_detail {
void native_rms_norm_rows(const float* input, const float* gamma, float* output, int n_cols, int n_rows,
                          int gamma_rows, float epsilon, void* stream);
}  // namespace strata::kernels::sycl_detail
