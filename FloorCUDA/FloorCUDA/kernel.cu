
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <stdio.h>

#define THREADCOUNT 1
#define BASE 10.0
#define EXP 9.0

cudaError_t addColumnsWithCuda(double * results, const double * const length);

__global__ void addColumnKernel(double * results, const double * const length)
{
    int i = threadIdx.x;

	results[i] = 0;

	double shortPartCount = i * (*length) + 1;
	double sqrt2PartCount = i;

	double maxLength = (i + 1) * (*length);
	while (shortPartCount <= maxLength) {
		if (sqrt2PartCount * sqrt2PartCount >= shortPartCount * shortPartCount * 2)
		{
			shortPartCount += 1;
			results[i] += sqrt2PartCount;
		}
		sqrt2PartCount++;
	}
}

int main()
{
	double fullLength = (double)pow(BASE, EXP);

	printf("Full length: %f\n", fullLength);

	double length = fullLength / THREADCOUNT;

	printf("Thread length: %f\n", length);

	double * results = (double *)malloc(THREADCOUNT * sizeof(double));

    // Add vectors in parallel.
    cudaError_t cudaStatus = addColumnsWithCuda(results, &length);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addColumnsWithCuda failed!");
        return 1;
    }

	double result = 1;
	for (int i = 0; i < THREADCOUNT; i++) {
		result += results[i];
	}
	printf("Result: %f\n", result);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addColumnsWithCuda(double * results, const double * const length)
{
	double * dev_results = 0;
	double * dev_length = 0;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_results, THREADCOUNT * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for dev_results!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_length, sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for dev_length!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_length, length, sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for dev_length!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addColumnKernel<<< 1, THREADCOUNT >>>(dev_results, dev_length);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		fprintf(stderr, "The last error was: %s\n", cudaGetErrorString(cudaGetLastError()));
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(results, dev_results, THREADCOUNT * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed results!");
        goto Error;
    }

Error:
    cudaFree(dev_results);
    cudaFree(dev_length);
    
    return cudaStatus;
}
