#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <bitset>

#include <utility>
#include <vector>

#include "BitSequence.cu"

using namespace std;

template<unsigned int K>
__host__ __device__ char compareSequences(BitSequence<K> * sequence1, BitSequence<K> * sequence2)
{
	int diff = 0;
	for (int j = 0; j < (K + 63) / 64; ++j)
	{
		unsigned long long int a, b, xor;
		a = *(sequence1->GetWord64(j));
		b = *(sequence2->GetWord64(j));
		xor = a ^ b;
		diff += xor == 0 ? 0 : (xor & (xor -1) ? 2 : 1);
		if (diff > 1)
		{
			return 0;
		}
	}
	return !!diff;
}
template<unsigned int N,unsigned int K>
void checkSequencesCPU(BitSequence<K> * sequence, BitSequence<N*(unsigned long long)N> *d_odata)
{
	unsigned long long numberOfComparisons = N * (N-1)/2;
	for(unsigned long long k = 0; k < numberOfComparisons; k+= 32)
	{
		unsigned int result = 0;
		for(int i = 0; i< 32 ; i++)
		{
			unsigned long long i1, i2;
			k2ij(k + i, &i1, &i2);
			result |= (unsigned int)(compareSequences<K>(sequence + i1, sequence + i2)) << i;
		}
		*(d_odata->GetWord32(k /32)) = result;
	}
}

__host__ __device__ void k2ij(unsigned long long  k, unsigned long long * i, unsigned long long  * j)
{
	//adding 1 to k to skip first result
	k += 1;
	*i = (unsigned long long)(ceill(0.5 * (-1 + sqrtl(1 + 8 * k))));
	//decreasing 1 from j , as we start from 0 not 1
	*j = (unsigned long long)(k - 0.5 * (*i) * ((*i) - 1)) - 1;
}

__host__ __device__ unsigned long long ij2k(unsigned long long i, unsigned long long j)
{
	return i * (i - 1) / 2 + j;
}

template<int N, int K>
__global__ void checkSequences(BitSequence<K> * d_sequence, BitSequence<N*N> *d_odata, unsigned long long offset = 0)
{
	unsigned long long i = threadIdx.x + 1024 * blockIdx.x + offset;
	unsigned long long i1, i2;
	k2ij(i, &i1, &i2);
	char res = compareSequences<K>(d_sequence + i1, d_sequence + i2);
	unsigned int bs = __ballot_sync(~0, res * (1 << i % 32));
	*(d_odata->GetWord32(i / 32)) = bs;
	//printf("Tid %d, i1 %d, i2 %d, res %d, bs %d\n", i, i1, i2, res, bs);
}

const unsigned int K = 10000; //Number of bits in one sequence
const unsigned int N = 100000; //Number of sequences
const unsigned int L = 10000000; //Number of blocks in one kernel call


ostream & operator<<(ostream & out, BitSequence<K> & sequence)
{
	for (int i = 0; i < K; ++i)
	{
		out << (short int)sequence.GetBit(i);
	}
	return out;
}



BitSequence<K> * Generate()
{
	srand(2018);

	BitSequence<K> * r = new BitSequence<K>[N];
	memset(r, 0, sizeof(BitSequence<K>)*N);

	for (int i = 0; i < N; i++)
	{
		int steps = rand() % K;
		for (int j = 0; j < steps; j++)
		{
			int index = rand() % K;
			char value = rand() % 2;
			r[i].SetBit(index, value);
		}
	}
	return r;
}

void printAsMatrix(const BitSequence<N*N> & sequence, ostream & stream)
{
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			if (j <= i)
			{
				cout << "  ";
			}
			else
			{
				cout << (short int)sequence.GetBit(ij2k(i,j))<< " ";
			}
		}
		cout << endl;
	}
}

void printAsMatrix(const vector<std::pair<int,int>>, ostream & stream)
{
	/*for (int j = 0; j < N; ++j)
	{
		for (int i = 1; j < N; ++i)
		{
			if (j <= i)
			{
				cout << "  ";
			}
			else
			{
				cout << sequence.GetBit(ij2k(i, j)) + '0' << " ";
			}
		}
		cout << endl;
	}*/
}

vector<pair<int, int>> findPairs(BitSequence<K> * h_sequence)
{
	cudaError_t cudaStatus;
	BitSequence<K> *d_sequence;
	cudaStatus = cudaMalloc((void**)&d_sequence, sizeof(BitSequence<K>)*N);
	/*for (int i = 0; i < N; ++i)
	{
		cout << i << ": " << *(h_sequence + i) << endl;
	}*/
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!  Do you have a CUDA-capable GPU installed?");
	}
	cudaStatus = cudaMemcpy(d_sequence, h_sequence, sizeof(BitSequence<K>)*N, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!  Do you have a CUDA-capable GPU installed?");
	}
	//Too big to keep on stack
	BitSequence<N*N> *d_odata, *h_odata_p, *h_odata_p2;
	h_odata_p = new BitSequence<N*N>;
	h_odata_p2 = new BitSequence<N*N>;
	BitSequence<N*N> & h_odata = *h_odata_p;
	BitSequence<N*N> & h_odata2 = *h_odata_p2;
	//printAsMatrix(h_odata, cout);
	cudaStatus = cudaMalloc(&d_odata, sizeof(BitSequence<N*N>));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMallocfailed!  Do you have a CUDA-capable GPU installed?");
	}

	cudaEvent_t start, stop; float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord( start, 0 );
	printf("Starting counting on GPU...\n");
	unsigned long long  offset = 0;
	unsigned long long nT = N * (N - 1) / 2;
	for(unsigned int i = 0; 1024*L*i < nT; ++i)
	{
		checkSequences<N, K> <<< L, 1024 >>> (d_sequence, d_odata, offset);
		offset += L*1024;
		cudaDeviceSynchronize();
	}
	checkSequences<N, K> <<<(nT%L), 1024 >>> (d_sequence, d_odata, offset);
	offset += (nT%L)*1024;
	cudaDeviceSynchronize();
	checkSequences<N, K> <<<1, nT - offset >>> (d_sequence, d_odata, offset);

	cudaDeviceSynchronize();

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );

	cudaEventElapsedTime( &time, start, stop );

	cudaEventDestroy( start );
	cudaEventDestroy( stop );
    printf("GPU Processing time: %f (ms)\n", time);

	cudaStatus = cudaMemcpy(&h_odata, d_odata, sizeof(h_odata), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		const char *err_str = cudaGetErrorString(cudaStatus);
		fprintf(stderr, "cudaMemcpy failed!  Do you have a CUDA-capable GPU installed?\n%s\n", err_str);
	}
	cudaDeviceSynchronize();
	//printAsMatrix(h_odata, cout);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cudaEventSynchronize(start);
	printf("Starting counting on CPU... \n");
	checkSequencesCPU<N, K>(h_sequence, &h_odata2);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    printf("CPU Processing time: %f (ms)\n", time);
	//printAsMatrix(h_odata2, cout);
	cout << "Comparison: " << endl;
	for (unsigned long long i = 0; i < N*(N - 1) / 2; ++i)
	{
		if (h_odata2.GetBit(i) != h_odata.GetBit(i))
		{
			unsigned long long i1, i2;
			k2ij(i, &i1, &i2);
			cout << "Difference on comparison number " << i << " (" << i1 << ", " << i2 << ") GPU " << (short int)h_odata.GetBit(i) << " CPU " << (short int)h_odata2.GetBit(i) << endl;
			unsigned int diff = 0;
			for (unsigned int j = 0; j < K; ++j)
			{
				diff += h_sequence[i1].GetBit(j) ^ h_sequence[i2].GetBit(j);
			}
			cout << "No of diffs: " << diff << endl;
		}
	}
	//printAsMatrix(h_odata, cout);
	// ZWRACANIE TABLICY DWÓJEK
	vector<pair<int, int>> result;
	
	for (unsigned long long k = 0; k < N * N; k++)
	{
		if (h_odata.GetBit(k))
		{
			unsigned long long i, j;
			k2ij(k, &i, &j);
			result.push_back(make_pair((int)i, (int)j));
		}
	}

	cudaFree(d_sequence);
	cudaFree(d_odata);
	delete h_odata_p;
	delete h_odata_p2;
	/*for (int i = 0; i < N*(N-1)/2; ++i)
	{
		int i1, i2;
		k2ij(i, &i1, &i2);
		cout << i1 << " " << i2 << ": " << (short int)h_odata.GetBit(i) << endl;
	}*/
	return result;
}

int main()
{
	printf("Starting sequence generation...\n");
	BitSequence<K>* sequence = Generate();
	printf("Ended sequence generation!\n");
	//f();
	// Add vectors in parallel.
	cudaError_t cudaStatus = cudaSuccess;
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	auto res = findPairs(sequence);

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
