
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <stdio.h>

#define THREADCOUNT 1000
#define BASE 10.0
#define EXP 9.0

cudaError_t addColumnsWithCuda(unsigned __int64 * results, const unsigned __int64 * const length, const double * const sqrt2);

__global__ void addColumnKernel(unsigned __int64 * results, const unsigned __int64 * const length, const double * const sqrt2)
{
    int i = threadIdx.x;

	results[i] = 0;

	unsigned __int64 maxLength = (i + 1) * *length;
	for (unsigned __int64 j = i * *length + 1; j <= maxLength; j++) {
		results[i] += ceil(j * *sqrt2);
	}
}

static double getSqrt(double x)
{
	double sqrt_x = sqrt(x);
	for (int i = 0; i < 10; ++i)
		sqrt_x = 0.5 * sqrt_x + 1 / sqrt_x;

	return sqrt_x;
}

int main()
{
	unsigned __int64 fullLength = (unsigned __int64)pow(BASE, EXP);

	printf("Full length: %llu\n", fullLength);

	unsigned __int64 length = fullLength / THREADCOUNT;

	printf("Thread length: %llu\n", length);

	unsigned __int64 * results = (unsigned __int64 *)malloc(THREADCOUNT * sizeof(unsigned __int64));
	double sqrt2 = 1.41421356237309504880168872420969807856967187537694807317667973799073247846210703885038753432764157273501384623091229702492483605585073721264412149709993583141322266592750559275579995050115278206057147010955997160597027453459686201472851741864088919860955232923048430871432145083976260362799525140798968725339654633180882964062061525835239505474575028775996172983557522033753185701135437460340849884716038689997069900481503054402779031645424782306849293691862158057846311159666871301301561856898723723528850926486124949771542183342042856860601468247207714358548741556570696776537202264854470158588016207584749226572260020855844665214583988939443709265918003113882464681570826301005948587040031864803421948972782906410450726368813137398552561173220402450912277002269411275736272804957381089675040183698683684507257993647290607629969413804756548237289971803268024744206292691248590521810044598421505911202494413417285314781058036033710773091828693147101711116839165817268894197587165821521282295184884720896946338628915628827659526351405422676532396946175112916024087155101351504553812875600526314680171274026539694702403005174953188629256313851881634780015693691768818523786840522878376293892143006558695686859645951555016447245098368960368873231143894155766510408839142923381132060524336294853170499157717562285497414389991880217624309652065642118273167262575395947172559346372386322614827426222086711558395999265211762526989175409881593486400834570851814722318142040704265090565323333984364578657967965192672923998753666172159825788602633636178274959942194037777536814262177387991945513972312740668983299898953867288228563786977496625199665835257761989393228453447356947949629521688914854925389047558288345260965240965428893945386466257449275563819644103169798330618520193793849400571563337205480685405758679996701213722394758214263065851322174088323829472876173936474678374319600015921888073478576172522118674904249773669292073110963697216089337086611567345853348332952546758516447107578486024636008344;

    // Add vectors in parallel.
    cudaError_t cudaStatus = addColumnsWithCuda(results, &length, &sqrt2);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addColumnsWithCuda failed!");
        return 1;
    }

	unsigned __int64 result = 1ull;
	for (int i = 0; i < THREADCOUNT; i++) {
		result += results[i];
	}
	printf("Result: %llu\n", result);

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
cudaError_t addColumnsWithCuda(unsigned __int64 * results, const unsigned __int64 * const length, const  double * const sqrt2)
{
	unsigned __int64 * dev_results = 0;
	unsigned __int64 * dev_length = 0;
	 double * dev_sqrt2;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_results, THREADCOUNT * sizeof(unsigned __int64));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for dev_results!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_length, sizeof(unsigned __int64));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for dev_length!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_sqrt2, sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for dev_sqrt2!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_length, length, sizeof(unsigned __int64), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for dev_length!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_sqrt2, sqrt2, sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for dev_sqrt2!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addColumnKernel<<< 1, THREADCOUNT >>>(dev_results, dev_length, dev_sqrt2);

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
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(results, dev_results, THREADCOUNT * sizeof(unsigned __int64), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed results!");
        goto Error;
    }

Error:
    cudaFree(dev_results);
    cudaFree(dev_length);
    cudaFree(dev_sqrt2);
    
    return cudaStatus;
}
