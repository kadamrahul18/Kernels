%%writefile load_inline.py
import torch
from torch.utils.cpp_extension import load_inline
import os

build_dir = '/kaggle/working/load_inline_cuda'
os.makedirs(build_dir, exist_ok=True)

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void square_matrix_kernel(const float* matrix, float* result, int num_rows, int num_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < num_rows && col < num_cols) {
        int i = row * num_cols + col;
        result[i] = matrix[i] * matrix[i];
    }
}

torch::Tensor square_matrix(torch::Tensor matrix) {
    TORCH_CHECK(matrix.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(matrix.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(matrix.dim() == 2, "Input tensor must be 2-dimensional");
    TORCH_CHECK(matrix.scalar_type() == torch::kFloat32, "Input tensor must be of type float32");

    const auto num_rows = matrix.size(0);
    const auto num_cols = matrix.size(1);

    auto result = torch::empty_like(matrix);

    dim3 threads_per_block(16, 16);

    dim3 blocks_per_grid(
        (num_cols + threads_per_block.x - 1) / threads_per_block.x,
        (num_rows + threads_per_block.y - 1) / threads_per_block.y
    );

    square_matrix_kernel<<<blocks_per_grid, threads_per_block>>>(
        matrix.data_ptr<float>(),
        result.data_ptr<float>(),
        num_rows,
        num_cols
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
    // cudaDeviceSynchronize();

    return result;
}
"""

cpp_source = "torch::Tensor square_matrix(torch::Tensor matrix);"

# Load the extension
square_matrix_extension = load_inline(
    name='square_matrix_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['square_matrix'],
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    build_directory=build_dir,
    verbose=True
)

a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device='cuda', dtype=torch.float32)
print("Input tensor a:")
print(a)

b = square_matrix_extension.square_matrix(a)
print("\\nResult from custom CUDA kernel:")
print(b)

print("\\nExpected result (a*a):")
print(a * a)

assert torch.allclose(b, a * a), "CUDA kernel result does not match a*a"
print("\\nVerification successful!")

#!python load_inline.py

#!ncu python load_inline.py