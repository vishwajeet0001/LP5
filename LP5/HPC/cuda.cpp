#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// Vector Addition Kernel
_global_ void vectorAdd(int *d_a, int *d_b, int *d_c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        d_c[i] = d_a[i] + d_b[i];
    }
}

// Matrix Multiplication Kernel
_global_ void matrixMultiplyKernel(float *a, float *b, float *c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0;
        for (int i = 0; i < N; i++) {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

// Vector Addition Class
class VectorAddition {
public:
    void performVectorAddition(int N) {
        int *a, *b, *c, *d;
        int *d_a, *d_b, *d_c;
        size_t size = N * sizeof(int);

        a = (int *)malloc(size);
        b = (int *)malloc(size);
        c = (int *)malloc(size);
        d = (int *)malloc(size);

        srand(time(NULL));
        for (int i = 0; i < N; i++) {
            a[i] = rand() % 100;
            b[i] = rand() % 100;
        }

        clock_t start_cpu = clock();
        for (int i = 0; i < N; i++) {
            c[i] = a[i] + b[i];
        }
        clock_t end_cpu = clock();
        double cpu_time = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;

        cudaMalloc((void **)&d_a, size);
        cudaMalloc((void **)&d_b, size);
        cudaMalloc((void **)&d_c, size);
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
        cudaEventRecord(stop);

        cudaMemcpy(d, d_c, size, cudaMemcpyDeviceToHost);
        cudaEventSynchronize(stop);
        float gpu_time = 0;
        cudaEventElapsedTime(&gpu_time, start, stop);

//verification  equality for vector addition 
        bool match = true;
        for (int i = 0; i < N; i++) {
            if (c[i] != d[i]) {
                match = false;
                break;
            }
        }

        printf("\nVector Addition Results:\n");
        printf("CPU Time: %.6f s\n", cpu_time);
        printf("GPU Time: %.6f ms\n", gpu_time);
        printf("Speedup Factor: %.2f\n", (cpu_time) * 1000 / gpu_time);
        printf("Arrays Match: %s\n\n", match ? "Yes" : "No");

        free(a); free(b); free(c); free(d);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};

// Matrix Multiplication Class
class MatrixMultiplier {
private:
    float *hostA, *hostB, *hostC, *hostD;
    float *devA, *devB, *devC;
    int size;
    float cpuTime, gpuTime;

public:
    MatrixMultiplier(int N) {
        size = N * N * sizeof(float);
        hostA = (float *)malloc(size);
        hostB = (float *)malloc(size);
        hostC = (float *)malloc(size);
        hostD = (float *)malloc(size);
        cudaMalloc((void **)&devA, size);
        cudaMalloc((void **)&devB, size);
        cudaMalloc((void **)&devC, size);
        cpuTime = gpuTime = 0.0;
    }

    ~MatrixMultiplier() {
        free(hostA);
        free(hostB);
        free(hostC);
        free(hostD);
        cudaFree(devA);
        cudaFree(devB);
        cudaFree(devC);
    }

    void initializeMatrices(int N) {
        for (int i = 0; i < N * N; i++) {
            hostA[i] = rand() % 100;
            hostB[i] = rand() % 100;
        }
    }

    void gpuMatrixMultiplication(int N) {
        cudaMemcpy(devA, hostA, size, cudaMemcpyHostToDevice);
        cudaMemcpy(devB, hostB, size, cudaMemcpyHostToDevice);

        dim3 dimBlock(16, 16);
        dim3 dimGrid((N + 15) / 16, (N + 15) / 16);

        clock_t tic = clock();
        matrixMultiplyKernel<<<dimGrid, dimBlock>>>(devA, devB, devC, N);
        cudaDeviceSynchronize();
        clock_t toc = clock();

        gpuTime = ((float)(toc - tic)) / CLOCKS_PER_SEC;
        cudaMemcpy(hostC, devC, size, cudaMemcpyDeviceToHost);
    }

    void cpuMatrixMultiplication(int N) {
        clock_t tic = clock();
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < N; k++) {
                    sum += hostA[i * N + k] * hostB[k * N + j];
                }
                hostD[i * N + j] = sum;
            }
        }
        clock_t toc = clock();
        cpuTime = ((float)(toc - tic)) / CLOCKS_PER_SEC;
    }

//Verifying equality for matrix mult
    bool verifyEquality(int N) {
        float tolerance = 1e-5;
        for (int i = 0; i < N * N; i++) {
            if (fabs(hostC[i] - hostD[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    void printResults() {
        printf("Matrix Multiplication Results:\n");
        printf("CPU Time: %f seconds\n", cpuTime);
        printf("GPU Time: %f seconds\n", gpuTime);
        if (gpuTime > 0) {
            printf("Speed-Up Factor: %.2f x\n", cpuTime / gpuTime);
        } else {
            printf("Speed-Up Factor: N/A (GPU time too small)\n");
        }
    }
};

// Main
int main() {
    int N = 2048;  // Predefined value of N (size of matrix or vector)

    // Perform Vector Addition
    VectorAddition vectorAdder;
    vectorAdder.performVectorAddition(N);

    // Perform Matrix Multiplication
    MatrixMultiplier matrixMultiplier(N);
    matrixMultiplier.initializeMatrices(N);
    matrixMultiplier.cpuMatrixMultiplication(N);
    matrixMultiplier.gpuMatrixMultiplication(N);
    bool success = matrixMultiplier.verifyEquality(N);
    matrixMultiplier.printResults();
    printf("Verification: %s\n", success ? "true" : "false");

    return 0;
}








//2 nd code with no speedup things
/*
#include <iostream>
#include <cuda.h>
using namespace std;

#define N 512  // Size for both vector and square matrix

// Vector addition kernel
_global_ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Matrix multiplication kernel
_global_ void matMul(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0;
        for (int k = 0; k < n; ++k)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

int main() {
    // ---------- Vector Addition ----------      memory allocate krna h CPU and GPU pe
    int sizeVec = N * sizeof(float);              Calculates total size in bytes for one vector of N elements-> yeah device ke allocation ke liye kr rhe h
    float *a, *b, *c, *d_a, *d_b, *d_c;           3-3 pointer host or device ke liye HOST->cpu and device is gpu
    a = new float[N]; b = new float[N]; c = new float[N];
    cudaMalloc(&d_a, sizeVec); cudaMalloc(&d_b, sizeVec); cudaMalloc(&d_c, sizeVec);

    for (int i = 0; i < N; i++) { a[i] = i; b[i] = 2 * i; }      input le rhe h a or b me

    cudaMemcpy(d_a, a, sizeVec, cudaMemcpyHostToDevice);       ------copy data from the CPU (host) to the GPU (device) before kernel execution------
    cudaMemcpy(d_b, b, sizeVec, cudaMemcpyHostToDevice);
    vectorAdd<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);   --vectorAdd<<<blocks, threads>>>---,, aur kernel me GPU memory pointers are passed
    cudaMemcpy(c, d_c, sizeVec, cudaMemcpyDeviceToHost);

    cout << "Vector Add: c[0] = " << c[0] << ", c[N-1] = " << c[N - 1] << endl;

    // ---------- Matrix Multiplication ----------
    int sizeMat = N * N * sizeof(float);
    float *A, *B, *C, *d_A, *d_B, *d_C;
    A = new float[N * N]; B = new float[N * N]; C = new float[N * N];
    cudaMalloc(&d_A, sizeMat); cudaMalloc(&d_B, sizeMat); cudaMalloc(&d_C, sizeMat);

    for (int i = 0; i < N * N; i++) { A[i] = 1.0f; B[i] = 2.0f; }

    cudaMemcpy(d_A, A, sizeMat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeMat, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);           //Each block has 16 × 16 = 256 threads.
    dim3 blocks(N / threads.x, N / threads.y);   //threads.x==>16
    matMul<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaMemcpy(C, d_C, sizeMat, cudaMemcpyDeviceToHost);

    cout << "Matrix Mul: C[0][0] = " << C[0] << ", C[N-1][N-1] = " << C[N * N - 1] << endl;

    // Cleanup
    delete[] a; delete[] b; delete[] c;
    delete[] A; delete[] B; delete[] C;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
*/
