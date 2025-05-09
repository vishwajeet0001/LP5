#include <iostream>
#include <cuda.h>
#include <chrono>  // For measuring execution time
using namespace std;

#define N 512  // Size for both vector and square matrix

// GPU Vector Addition kernel
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate global index
    if (i < n) {
        c[i] = a[i] + b[i];  // Perform the addition
    }
}

// CPU Vector Addition
void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];  // Perform the addition
    }
}

int main() {
    int sizeVec = N * sizeof(float);              // Size of one vector of N elements
    float *a, *b, *c, *d_a, *d_b, *d_c;           // Pointers for host and device (CPU and GPU)
    a = new float[N]; b = new float[N]; c = new float[N];
    cudaMalloc(&d_a, sizeVec); cudaMalloc(&d_b, sizeVec); cudaMalloc(&d_c, sizeVec);

    // Initialize input vectors a and b
    for (int i = 0; i < N; i++) { 
        a[i] = i; 
        b[i] = 2 * i; 
    }

    // ---------- CPU Vector Addition ----------
    auto start_cpu = chrono::high_resolution_clock::now();
    vectorAddCPU(a, b, c, N);  // Perform vector addition on CPU
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<float> duration_cpu = end_cpu - start_cpu;
    cout << "CPU Vector Add: c[0] = " << c[0] << ", c[N-1] = " << c[N - 1] << endl;
    cout << "CPU Execution Time: " << duration_cpu.count() << " seconds" << endl;

    // ---------- GPU Vector Addition ----------
    // Copy data from host to device
    cudaError_t err = cudaMemcpy(d_a, a, sizeVec, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "CUDA error in cudaMemcpy (a): " << cudaGetErrorString(err) << endl;
        return -1;
    }

    err = cudaMemcpy(d_b, b, sizeVec, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "CUDA error in cudaMemcpy (b): " << cudaGetErrorString(err) << endl;
        return -1;
    }

    // Launch GPU kernel for vector addition
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;  // Number of blocks needed

    auto start_gpu = chrono::high_resolution_clock::now();
    vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);  // Launch GPU kernel
    err = cudaGetLastError();  // Check for kernel launch errors
    if (err != cudaSuccess) {
        cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    // Copy result from device to host
    err = cudaMemcpy(c, d_c, sizeVec, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cerr << "CUDA error in cudaMemcpy (c): " << cudaGetErrorString(err) << endl;
        return -1;
    }

    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<float> duration_gpu = end_gpu - start_gpu;
    cout << "GPU Vector Add: c[0] = " << c[0] << ", c[N-1] = " << c[N - 1] << endl;
    cout << "GPU Execution Time: " << duration_gpu.count() << " seconds" << endl;

    // Calculate Speedup (CPU / GPU)
    float speedup = duration_cpu.count() / duration_gpu.count();
    cout << "Speedup (CPU / GPU): " << speedup << "x" << endl;

    // Cleanup
    delete[] a; delete[] b; delete[] c;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}

//!nvcc -arch=sm_70 -O3 mainc.cu -o mainc try sm->50,60