#include <torch/extension.h>
torch::Tensor mspe_forward_cuda(torch::Tensor pred_log, torch::Tensor y_log, double eps);
torch::Tensor mspe_backward_cuda(torch::Tensor pred_log, torch::Tensor y_log, double eps);

torch::Tensor mspe_forward(torch::Tensor pred_log, torch::Tensor y_log, double eps) {
  TORCH_CHECK(pred_log.is_cuda() && y_log.is_cuda(), "Tensors must be CUDA");
  TORCH_CHECK(pred_log.sizes() == y_log.sizes(), "Size mismatch");
  TORCH_CHECK(pred_log.dtype() == torch::kFloat32, "Use float32");
  return mspe_forward_cuda(pred_log, y_log, eps);
}

torch::Tensor mspe_backward(torch::Tensor pred_log, torch::Tensor y_log, double eps) {
  TORCH_CHECK(pred_log.is_cuda() && y_log.is_cuda(), "Tensors must be CUDA");
  return mspe_backward_cuda(pred_log, y_log, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mspe_forward", &mspe_forward, "MSPE forward (CUDA)");
  m.def("mspe_backward", &mspe_backward, "MSPE backward (CUDA)");
}
