#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Size for matrix (N x N)
#define M 1000000  // Size for vector (M)

using namespace std;

// CUDA kernel for vector addition
__global__ void vector_add(int *A, int *B, int *C, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// CUDA kernel for matrix multiplication
__global__ void matrix_multiply(int *A, int *B, int *C, int n) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    // Vector Addition
    int *A_vec, *B_vec, *C_vec;
    int *d_A_vec, *d_B_vec, *d_C_vec;

    // Allocate memory for host vectors
    A_vec = new int[M];
    B_vec = new int[M];
    C_vec = new int[M];

    // Initialize vectors A and B
    for (int i = 0; i < M; i++) {
        A_vec[i] = i;
        B_vec[i] = i * 2;
    }

    // Allocate memory for device vectors
    cudaMalloc(&d_A_vec, M * sizeof(int));
    cudaMalloc(&d_B_vec, M * sizeof(int));
    cudaMalloc(&d_C_vec, M * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A_vec, A_vec, M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_vec, B_vec, M * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel for vector addition
    int blockSize_vec = 256;
    int numBlocks_vec = (M + blockSize_vec - 1) / blockSize_vec;
    vector_add<<<numBlocks_vec, blockSize_vec>>>(d_A_vec, d_B_vec, d_C_vec, M);

    // Copy result from device to host
    cudaMemcpy(C_vec, d_C_vec, M * sizeof(int), cudaMemcpyDeviceToHost);
    cout << "Vector Addition: C[0] = " << C_vec[0] << ", C[M-1] = " << C_vec[M-1] << endl;

    // Matrix Multiplication
    int *A_mat, *B_mat, *C_mat;
    int *d_A_mat, *d_B_mat, *d_C_mat;

    // Allocate memory for host matrices
    A_mat = new int[N * N];
    B_mat = new int[N * N];
    C_mat = new int[N * N];

    // Initialize matrices A and B
    for (int i = 0; i < N * N; i++) {
        A_mat[i] = 1;  // Fill A with 1s
        B_mat[i] = 2;  // Fill B with 2s
    }

    // Allocate memory for device matrices
    cudaMalloc(&d_A_mat, N * N * sizeof(int));
    cudaMalloc(&d_B_mat, N * N * sizeof(int));
    cudaMalloc(&d_C_mat, N * N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A_mat, A_mat, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_mat, B_mat, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel for matrix multiplication
    dim3 blockSize_mat(16, 16);
    dim3 numBlocks_mat((N + blockSize_mat.x - 1) / blockSize_mat.x, (N + blockSize_mat.y - 1) / blockSize_mat.y);
    matrix_multiply<<<numBlocks_mat, blockSize_mat>>>(d_A_mat, d_B_mat, d_C_mat, N);

    // Copy result from device to host
    cudaMemcpy(C_mat, d_C_mat, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    cout << "Matrix Multiplication: C[0][0] = " << C_mat[0] << ", C[N-1][N-1] = " << C_mat[N*N-1] << endl;

    // Free memory
    delete[] A_vec;
    delete[] B_vec;
    delete[] C_vec;
    delete[] A_mat;
    delete[] B_mat;
    delete[] C_mat;

    cudaFree(d_A_vec);
    cudaFree(d_B_vec);
    cudaFree(d_C_vec);
    cudaFree(d_A_mat);
    cudaFree(d_B_mat);
    cudaFree(d_C_mat);

    return 0;
}
