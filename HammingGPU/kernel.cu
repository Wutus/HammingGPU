#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

#include "BitSequence.cu"

using namespace std;
using namespace thrust;



__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

template<unsigned int K>
__device__ char compareSequences(BitSequence<K> * sequence1, BitSequence<K> * sequence2)
{
	int diff = 0;
	for (int j = 0; j < (K + 63) / 64; ++j)
	{
		unsigned long long int a, b, xor;
		a = *(sequence1->GetWord64(j));
		b = *(sequence2->GetWord64(j));
		xor = a ^ b;
		diff += xor == 0 ? 0 : (xor & (xor - 1) ? 2 : 1);
		if (diff > 1)
		{
			return 0;
		}
	}
	return !!diff;
}

__host__ __device__ void k2ij(int k, int * i, int * j)
{
	*i = (int)(ceilf(0.5*(-1 + sqrtf(1 + 8 * k))));
	*j = (int)(k - 0.5 * (*i) * ((*i) - 1));
}

template<int N, int K>
__global__ void checkSequences(BitSequence<K> * d_sequence, BitSequence<N*N> *d_odata)
{
	int i = threadIdx.x + 1024 * blockIdx.x;
	int i1 = (int)(ceilf(0.5*(-1 + sqrtf(1 + 8 * i))));
	int i2 = (int)(i - 0.5 * i1 * (i1 - 1));
	char res = compareSequences<K>(d_sequence + i1, d_sequence + i2);
	d_odata->SetBit(i, res);
	unsigned int bs = __ballot_sync(-1, res * (1 << i % 32));
	*(d_odata->GetWord32(i/32)) = bs;
	printf("Tid %d, i1 %d, i2 %d, res %d, bs %d\n", i, i1, i2, res, bs);
}

const unsigned int K = 64;
const unsigned int N = 10;

BitSequence<K> * Generate()
{
	srand(2018);

	BitSequence<K> * r = new BitSequence<K>[N];
	memset(r, 0, sizeof(BitSequence<K>)*N);
	
	/*for (int i = 0; i < N; i++)
	{
		int steps = rand() % K;
		for (int j = 0; j < steps; j++)
		{
			int index = rand() % K;
			char value = rand() % 2;
			r[i].SetBit(index, value);
		}
	}*/

	r[0].SetBit(0, 1);
	r[1].SetBit(1, 1);
	r[2].SetBit(2, 1);
	return r;
}

unsigned int *findPairs(BitSequence<K> * h_sequence)
{
	cudaError_t cudaStatus;
	BitSequence<K> *d_sequence;
	cudaStatus = cudaMalloc((void**)&d_sequence, sizeof(BitSequence<K>)*N);
	for (int i = 0; i < N; ++i)
	{
		printf("%d: %d\n", i, *(h_sequence[i].GetWord64(0)));
	}
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!  Do you have a CUDA-capable GPU installed?");
	}
	cudaStatus = cudaMemcpy(d_sequence, h_sequence, sizeof(BitSequence<K>)*N, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!  Do you have a CUDA-capable GPU installed?");
	}
	BitSequence<N*N> *d_odata, h_odata;
	cudaMalloc(&d_odata, sizeof(BitSequence<N*N>));
	//addKernel <<<1, 1 >>> ((int*)d_sequence, (int*)d_sequence,(int*)d_sequence);
	checkSequences<N, K> <<< 1, N*(N-1)/2 >> > (d_sequence, d_odata);
	cudaDeviceSynchronize();
	cudaMemcpy(&h_odata, d_odata, sizeof(h_odata), cudaMemcpyDeviceToHost);
	for (int i = 0; i < N*(N-1)/2; ++i)
	{
		int i1, i2;
		k2ij(i, &i1, &i2);
		cout << i1 << " " << i2 << ": " << (short int)h_odata.GetBit(i) << endl;
	}
	return NULL;
}

int main()
{
	//f();
	// Add vectors in parallel.
	cudaError_t cudaStatus = cudaSuccess;
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	auto res = findPairs(Generate());
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
/*cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

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
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}*/
