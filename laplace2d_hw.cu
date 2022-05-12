/* Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <cuda.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <iostream>

// This is a little wrapper that checks for error codes returned by CUDA API calls
#define cudaCheck(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__global__ void my_stencil_kernel(double *A, double *Anew, int imax, int jmax, double *d_x)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int id = threadIdx.x + threadIdx.y * blockDim.x;
    __shared__ double local[64];
    if (i >= 1 && i < imax + 1 && j >= 1 && j < jmax + 1)
    {
        if (id < (imax * jmax)){
            local[threadIdx.x + threadIdx.y] = A[id];
        }
        else{
            local[threadIdx.x + threadIdx.y] = 0.0;
        }
        Anew[(j) * (imax + 2) + i] = 0.25f * (A[(j) * (imax + 2) + i + 1] + A[(j) * (imax + 2) + i - 1] + A[(j - 1) * (imax + 2) + i] + A[(j + 1) * (imax + 2) + i]);
        local[threadIdx.x + threadIdx.y] = Anew[id];
        for (int d = blockDim.x * blockDim.y >> 1; d >= 1; d >>= 1)
        {
            __syncthreads();
            // (j) * (imax + 2) + i, fabs(Anew[(j) * (imax + 2) + i] - A[(j) * (imax + 2) + i])
            if ((threadIdx.x + threadIdx.y) < d) local[threadIdx.x + threadIdx.y] = fabs(local[threadIdx.x + threadIdx.y+d] - local[threadIdx.x + threadIdx.y+d]);
        }

        if (threadIdx.x == 0)
        {
            atomicMax(d_x, local[0]);
        }
    }
}

__global__ void my_copy_kernel(double *A, double *Anew, int imax, int jmax)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= 1 && i < imax + 1 && j >= 1 && j < jmax + 1)
        A[(j) * (imax + 2) + i] = Anew[(j) * (imax + 2) + i];
}

int main(int argc, char **argv)
{
    // Size along y
    int jmax = 4094;
    // Size along x
    int imax = 4094;
    int iter_max = 1000;

    const double pi = 2.0 * asin(1.0);
    const double tol = 1.0e-5;
    double error = 1.0;

    double *A;
    double *Anew;
    double *y0;

    A = (double *)malloc((imax + 2) * (jmax + 2) * sizeof(double));
    Anew = (double *)malloc((imax + 2) * (jmax + 2) * sizeof(double));
    y0 = (double *)malloc((imax + 2) * sizeof(double));

    memset(A, 0, (imax + 2) * (jmax + 2) * sizeof(double));

    // set boundary conditions
    for (int i = 0; i < imax + 2; i++)
        A[(0) * (imax + 2) + i] = 0.0;

    for (int i = 0; i < imax + 2; i++)
        A[(jmax + 1) * (imax + 2) + i] = 0.0;

    for (int j = 0; j < jmax + 2; j++)
    {
        y0[j] = sin(pi * j / (jmax + 1));
        A[(j) * (imax + 2) + 0] = y0[j];
    }

    for (int j = 0; j < imax + 2; j++)
    {
        y0[j] = sin(pi * j / (jmax + 1));
        A[(j) * (imax + 2) + imax + 1] = y0[j] * exp(-pi);
    }

    printf("Jacobi relaxation Calculation: %d x %d mesh\n", imax + 2, jmax + 2);

    // double t1 = omp_get_wtime();
    int iter = 0;

    for (int i = 1; i < imax + 2; i++)
        Anew[(0) * (imax + 2) + i] = 0.0;

    for (int i = 1; i < imax + 2; i++)
        Anew[(jmax + 1) * (imax + 2) + i] = 0.0;

    for (int j = 1; j < jmax + 2; j++)
        Anew[(j) * (imax + 2) + 0] = y0[j];

    for (int j = 1; j < jmax + 2; j++)
        Anew[(j) * (imax + 2) + jmax + 1] = y0[j] * expf(-pi);

    double *d_A;
    double *d_Anew;
    double *d_x;
    cudaCheck(cudaMalloc(&d_A, (imax + 2) * (jmax + 2) * sizeof(double)));
    cudaCheck(cudaMemcpy(d_A, A, (imax + 2) * (jmax + 2) * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMalloc(&d_Anew, (imax + 2) * (jmax + 2) * sizeof(double)));
    cudaCheck(cudaMemcpy(d_Anew, Anew, (imax + 2) * (jmax + 2) * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMalloc(&d_x, (imax + 2) * (jmax + 2) * sizeof(double)));
    cudaCheck(cudaMemset(d_x, 0, (imax + 2) * (jmax + 2) * sizeof(double)));

    while (error > tol && iter < iter_max)
    {
        error = 0.0;

        dim3 block(16, 4);
        dim3 grid((imax + 2 - 1) / 16 + 1, (jmax + 2 - 1) / 4 + 1);
        cudaCheck(cudaDeviceSynchronize());
        my_stencil_kernel<<<grid, block>>>(d_A, d_Anew, imax, jmax, d_x);
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaMemcpy(&error, d_x, sizeof(double), cudaMemcpyDeviceToHost));
        // No stencil accesses to Anew, no halo exchange necessary
        my_copy_kernel<<<grid, block>>>(d_A, d_Anew, imax, jmax);
        if (iter % 100 == 0)
            printf("%5d, %0.6f\n", iter, error);

        iter++;
    }

    // double runtime = omp_get_wtime()-t1;
    double runtime = 1;
    printf(" total: %f s\n", runtime);
}
