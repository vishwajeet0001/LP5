#include <iostream>
#include <cuda.h>
#include <chrono>  // For measuring execution time
using namespace std;

#define N 512  // Size for both square matrices

// GPU Matrix Multiplication kernel
__global__ void matMul(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];  // Perform the multiplication for C[row][col]
        }
        C[row * n + col] = sum;  // Store the result in the output matrix C
    }
}

// CPU Matrix Multiplication
void matMulCPU(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];  // Multiply and sum
            }
        }
    }
}

int main() {
    int sizeMat = N * N * sizeof(float);  // Size of one matrix (N x N)
    float *A, *B, *C, *d_A, *d_B, *d_C;  // Pointers for host and device (CPU and GPU)
    A = new float[N * N]; B = new float[N * N]; C = new float[N * N];
    cudaMalloc(&d_A, sizeMat); cudaMalloc(&d_B, sizeMat); cudaMalloc(&d_C, sizeMat);

    // Initialize matrices A and B with some values (e.g., sequential values for simplicity)
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;  // Initialize A with 1s
        B[i] = 2.0f;  // Initialize B with 2s
    }

    // ---------- CPU Matrix Multiplication ----------
    auto start_cpu = chrono::high_resolution_clock::now();
    matMulCPU(A, B, C, N);  // Perform matrix multiplication on CPU
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<float> duration_cpu = end_cpu - start_cpu;
    cout << "CPU Matrix Mul: C[0][0] = " << C[0] << ", C[N-1][N-1] = " << C[N * N - 1] << endl;
    cout << "CPU Execution Time: " << duration_cpu.count() << " seconds" << endl;

    // ---------- GPU Matrix Multiplication ----------
    // Copy data from host to device
    cudaError_t err = cudaMemcpy(d_A, A, sizeMat, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "CUDA error in cudaMemcpy (A): " << cudaGetErrorString(err) << endl;
        return -1;
    }

    err = cudaMemcpy(d_B, B, sizeMat, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "CUDA error in cudaMemcpy (B): " << cudaGetErrorString(err) << endl;
        return -1;
    }

    // Launch GPU kernel for matrix multiplication
    int threadsPerBlock = 16;  // Threads per block
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;  // Number of blocks needed

    dim3 threads(threadsPerBlock, threadsPerBlock);  // Define the block size as 16x16 threads
    dim3 grid(blocks, blocks);  // Grid size is based on the number of blocks required

    auto start_gpu = chrono::high_resolution_clock::now();
    matMul<<<grid, threads>>>(d_A, d_B, d_C, N);  // Launch GPU kernel
    err = cudaGetLastError();  // Check for kernel launch errors
    if (err != cudaSuccess) {
        cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    // Copy result from device to host
    err = cudaMemcpy(C, d_C, sizeMat, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cerr << "CUDA error in cudaMemcpy (C): " << cudaGetErrorString(err) << endl;
        return -1;
    }

    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<float> duration_gpu = end_gpu - start_gpu;
    cout << "GPU Matrix Mul: C[0][0] = " << C[0] << ", C[N-1][N-1] = " << C[N * N - 1] << endl;
    cout << "GPU Execution Time: " << duration_gpu.count() << " seconds" << endl;

    // Calculate Speedup (CPU / GPU)
    float speedup = duration_cpu.count() / duration_gpu.count();
    cout << "Speedup (CPU / GPU): " << speedup << "x" << endl;

    // Cleanup
    delete[] A; delete[] B; delete[] C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}
