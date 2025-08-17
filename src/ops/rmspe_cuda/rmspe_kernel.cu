#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename T>
__global__ void mspe_forward_kernel(const T* __restrict__ pred_log,
                                    const T* __restrict__ y_log,
                                    T eps, int64_t N, T* out_sum) {
    
    /* Calculate loss for a batch of predictions on the forward pass */
    /*
    Grid-stride loop: Efficiently processes large arrays with good memory patterns
    Block reduction: Reduces communication overhead - only one value per block goes to global memory
    Atomic addition: Safely combines results from all blocks
    */
    T local = 0;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        T y  = expf(y_log[i]) - eps; // get actual y
        T p  = expf(pred_log[i]) - eps; // get actual pred
        T e  = (p - y) / (y + eps); // percentage error
        local += e * e; // accumulate squared error
    }
    // block reduction
    extern __shared__ T sm[];
    sm[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x/2; s>0; s>>=1) { if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(out_sum, sm[0]);
}

template <typename T>
__global__ void mspe_backward_kernel(const T* __restrict__ pred_log,
                                     const T* __restrict__ y_log,
                                     T eps, int64_t N, T invN, T* __restrict__ grad) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    T y  = expf(y_log[i]) - eps;
    T p  = expf(pred_log[i]) - eps;
    // Chain rule to get derivative of mspe w.r.t actual y
    // d/dp MSPE_i = 2*(p - y)/(y+eps)^2 * (1/N)
    T d_mspe_dp = 2.f * (p - y) * invN / ((y + eps)*(y + eps));
    // dp/d(pred_log) = p
    grad[i] = d_mspe_dp * p;
  }
}

torch::Tensor mspe_forward_cuda(torch::Tensor pred_log, torch::Tensor y_log, double eps) {
  auto N = pred_log.numel();
  auto out = torch::zeros({1}, pred_log.options());
  int threads = 256, blocks = (N + threads - 1) / threads;
  size_t shmem = threads * sizeof(float);
  mspe_forward_kernel<float><<<blocks, threads, shmem>>>(
      pred_log.data_ptr<float>(), y_log.data_ptr<float>(), (float)eps, N, out.data_ptr<float>());
  // divide by N (mean)
  out /= (float)N;
  return out.squeeze();
}

torch::Tensor mspe_backward_cuda(torch::Tensor pred_log, torch::Tensor y_log, double eps) {
  auto grad = torch::empty_like(pred_log);
  auto N = pred_log.numel();
  int threads = 256, blocks = (N + threads - 1) / threads;
  float invN = 1.f / (float)N;
  mspe_backward_kernel<float><<<blocks, threads>>>(
      pred_log.data_ptr<float>(), y_log.data_ptr<float>(), (float)eps, N, invN, grad.data_ptr<float>());
  return grad;
}
