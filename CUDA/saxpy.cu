#include<stdio.h>
#include<stdlib.h>
#include<math.h>

__global__ void saxpy(int n, float a, float* x, float* y){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < n){
        y[i] = a*x[i] + y[i];
    }
}

int main(void){
    int N = 1 << 20;

    float *x, *y;
    float *d_x, *d_y;

    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));

    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));

    for(int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    saxpy<<<blocksPerGrid, threadsPerBlock>>>(N, 2.0f, d_x, d_y);

    cudaDeviceSynchronize();

    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for(int i = 0; i < N; i++){
        maxError = max(maxError, abs(y[i] - 4.0f));
    }
    printf("Max Error: %f\n", maxError);

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

    return 0;
}