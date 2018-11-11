
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "BitSerie.h"

using namespace thrust;

const unsigned int K = 1000;
const unsigned int N = 1000;

host_vector<host_vector<unsigned int>> findPairs(const host_vector<BitSequence<K>> & h_sequence);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

template<int N, int K>
__global__ void checkSequence(device_vector<BitSequence<K>> & d_sequence, BitSequence<N*N> *d_odata)
{
	int i = threadIdx.x;
	int i1 = i / 8;
	int i2 = i % 8;
	int diff = 0;
	for (int j = 0; j < (K+63)/64; ++j)
	{
		unsigned long long int xor = (*(d_sequence[i1].GetWord64(j))) ^ (*(d_sequence[i2].GetWord64(j)));
		while(xor > 0)
		{
			diff += xor & 1;
			xor >>= 1;
		}
		if (diff > 1)
		{
			d_odata->SetBit(i, 0);
		}
	}
	d_odata->SetBit(i, diff);
}

int main()
{
    // Add vectors in parallel.
	cudaError_t cudaStatus = cudaSuccess;
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	auto res = findPairs(host_vector<BitSequence<K>>(N));
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

host_vector<host_vector<unsigned int>> findPairs(const host_vector<BitSequence<K>> & h_sequence)
{
	device_vector<BitSequence<K>> d_sequence(h_sequence);
	BitSequence<N*N> *d_odata, h_odata;
	cudaMalloc(&d_odata, sizeof(BitSequence<N*N>));
	checkSequence<N,K> <<< N, 1 >>>(d_sequence, d_odata);
	cudaDeviceSynchronize();
	cudaMemcpy(&h_odata, d_odata, sizeof(h_odata), cudaMemcpyDeviceToHost);
	host_vector<host_vector<unsigned int>> h_res;
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			if (d_odata->GetBit(i*N + j))
				h_res.push_back(host_vector<unsigned int>{i, j});
		}
	}
	return h_res;
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
