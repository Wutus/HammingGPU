#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>

using namespace std;

//#define ITER_GPU

#define CHECK_ERRORS(status) do{\
	if(cudaSuccess != status) {\
		fprintf(stderr, "Cuda Error in %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(status));\
	}\
}while(0)

#define CHECK_ERRORS_FORMAT(status, format, ...) do{\
	if(cudaSuccess != status) {\
		fprintf(stderr, "Cuda Error in %s:%d - %s", __FILE__, __LINE__, cudaGetErrorString(status));\
		fprintf(stderr, format, __VA_ARGS__);\
		fprintf(stderr, "\n");\
	}\
}while(0)

#define K 10000 //Number of bits in one sequence
#define N 100000 //Number of sequences
#define L ((N*(N - 1)) / 2) //Number of comparisons
#define B 100000 //Number of maximum blocks per call

template<unsigned long long k>
class BitSequence;
class CudaTimer;

__global__ void checkSequencesGPU(BitSequence<K> * d_sequence, BitSequence<L> *d_odata, unsigned long long offset = 0);
__host__ __device__ char compareSequences(BitSequence<K> * sequence1, BitSequence<K> * sequence2);
__host__ __device__ void k2ij(unsigned long long k, unsigned int * i, unsigned int  * j);
__host__ __device__ unsigned long long ij2k(unsigned int i, unsigned int j);
void checkSequencesCPU(BitSequence<K> * sequence, BitSequence<L> * odata);
void PrintComparison(const BitSequence<K> & gpu_sequence, const BitSequence<K> & cpu_sequence);
bool ComparePairs(const vector<pair<int, int> > & gpu_result, const vector<pair<int, int> > & cpu_result);

ostream & operator<<(ostream & out, BitSequence<K> & sequence);
BitSequence<K> * Generate();
vector<pair<int, int>> ToPairVector(const BitSequence<L> & result_sequence);
void printAsMatrix(const BitSequence<L> & sequence, ostream & stream);

vector<pair<int, int> > findPairsGPU(BitSequence<K> * h_sequence);
vector<pair<int, int> > findPairsCPU(BitSequence<K> * sequence);

int main()
{
	cudaError_t cudaStatus;
	printf("Starting sequence generation...\n");
	BitSequence<K>* sequence = Generate();
	printf("Ended sequence generation!\n");

	auto gpuRes = findPairsGPU(sequence);
	auto cpuRes = findPairsCPU(sequence);
	ComparePairs(gpuRes, cpuRes);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

template<unsigned long long k>
class BitSequence
{
public:
	__host__ BitSequence()
	{
	}
	__host__ BitSequence(char array[])
	{
		cudaMemcpy(this->array, array, arSize*8, cudaMemcpyHostToHost);
	}
	__host__ __device__ inline char GetBit(unsigned long long index) const
	{
		return (array[index / 64] >> (index % 64)) & 1;
	}
	__host__ __device__ inline void SetBit(unsigned long long index, char value)
	{
		array[index / 64] = (array[index / 64] & (~(1ull << (index % 64)))) | ((!!value) << (index % 64));
	}
	__host__ __device__ inline unsigned int *GetWord32(unsigned long long word_index)
	{
		return ((unsigned int*)array) + word_index);
	}
	__host__ __device__ inline unsigned long long *GetWord64(unsigned long long word_index)
	{
		return (array + word_index);
	}
	static const unsigned long long arSize = (k + 63) / 64;
private:
	unsigned long long array[arSize];
};

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

__host__ __device__ void k2ij(unsigned long long k, unsigned int * i, unsigned int  * j)
{
	*i = (unsigned int)ceil((0.5 * (-1 + sqrtl(1 + 8 * (k + 1)))));
	*j = (unsigned int)((k + 1) - 0.5 * (*i) * ((*i) - 1)) - 1;
}

__host__ __device__ unsigned long long ij2k(unsigned int i, unsigned int j)
{
	return ((unsigned long long)i) * (i - 1) / 2 + j;
}

void checkSequencesCPU(BitSequence<K> * sequence, BitSequence<L> * odata)
{
	unsigned long long numberOfComparisons = L;
	for (unsigned long long k = 0; k < numberOfComparisons; k += 32)
	{
		unsigned int result = 0;
		for (int i = 0; i < 32; i++)
		{
			unsigned int i1, i2;
			k2ij(k + i, &i1, &i2);
			result |= (unsigned int)(compareSequences(sequence + i1, sequence + i2)) << i;
		}
		*(odata->GetWord32(k / 32)) = result;
	}
}

class CudaTimer
{
public:
	CudaTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		started = false;
	}

	~CudaTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		started = true;
		cudaEventRecord(start);
		cudaEventSynchronize(start);
	}

	float Stop()
	{
		if (!started)
			return -1.0f;
		float ms;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms, start, stop);
		started = false;
		return ms;
	}
private:
	bool started;
	cudaEvent_t start, stop;
};

void PrintComparison(const BitSequence<K> & gpu_sequence, const BitSequence<K> & cpu_sequence)
{
	for (unsigned long long i = 0; i < N*(N - 1) / 2; ++i)
	{
		if (cpu_sequence.GetBit(i) != gpu_sequence.GetBit(i))
		{
			unsigned int i1, i2;
			k2ij(i, &i1, &i2);
			cout << "Difference on comparison number " << i << " (" << i1 << ", " << i2 << ") GPU " << (short int)gpu_sequence.GetBit(i) << " CPU " << (short int)cpu_sequence.GetBit(i) << endl;
		}
	}
}

bool ComparePairs(const vector<pair<int, int> > & gpu_result, const vector<pair<int, int> > & cpu_result)
{
	unsigned long long gsize = gpu_result.size(), csize = cpu_result.size();
	unsigned long long n = gsize < csize ? gsize : csize;
	const vector<pair<int, int> > & gv = gsize < csize ? cpu_result : gpu_result;
	bool equal = true;

	if (gsize != csize)
	{
		cout << "Number of elements is not equal (GPU: " << gsize << ", CPU: " << csize << ") !" << endl;
		equal = false;
	}

	int i;
	for (i = 0; i < n; ++i)
	{
		if (gpu_result[i] != cpu_result[i])
		{
			cout << "Difference on " << i << ": GPU: (" << gpu_result[i].first << ", " << gpu_result[i].second << ") CPU: ("
				<< cpu_result[i].first << ", " << cpu_result[i].second << ")";
			equal = false;
		}
	}
	if (csize != gsize)
	{
		cout << "Rest pairs on " << ((csize < gsize) ? "CPU" : "GPU") << ":" << endl;
		for (; i < gv.size(); ++i)
		{
			cout << "(" << gv[i].first << ", " << gv[i].second << ")" << endl;
		}
	}
	if (equal)
	{
		cout << "Results are the same" << endl;
	}
	return equal;
}

__global__ void checkSequencesGPU(BitSequence<K> * d_sequence, BitSequence<L> *d_odata, unsigned long long offset = 0)
{
	unsigned long long i = threadIdx.x + blockIdx.x * blockDim.x + offset;
	unsigned int i1, i2;
	k2ij(i, &i1, &i2);
	i2 = compareSequences(d_sequence + i1, d_sequence + i2);
	i1 = __ballot(i2);
	*(d_odata->GetWord32(i / 32)) = i1;
}

ostream & operator<<(ostream & out, BitSequence<K> & sequence)
{
	for (unsigned long long i = 0; i < K; ++i)
	{
		out << (short int)sequence.GetBit(i);
	}
	return out;
}

BitSequence<K> * Generate()
{
	srand(2018);

	BitSequence<K> * r = new BitSequence<K>[N];

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < K / 32; j++)
		{
			*(r[i].GetWord32(j)) = rand() + rand()*RAND_MAX;
		}
	}
	return r;
}

vector<pair<int, int>> ToPairVector(const BitSequence<L> & result_sequence)
{
	vector<pair<int, int>> result;
	for (unsigned long long k = 0; k < L; k++)
	{
		if (result_sequence.GetBit(k))
		{
			unsigned int i, j;
			k2ij(k, &i, &j);
			result.push_back(make_pair(i, j));
		}
	}

	return result;
}

void printAsMatrix(const BitSequence<L> & sequence, ostream & stream)
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
				cout << (short int)sequence.GetBit(ij2k(i, j)) << " ";
			}
		}
		cout << endl;
	}
}

vector<pair<int, int> > findPairsGPU(BitSequence<K> * h_sequence)
{
	BitSequence<K> *d_idata;
	BitSequence<L> *h_odata, *d_odata;
	CudaTimer timerCall, timerMemory;
	float xtime, xmtime;
	h_odata = new BitSequence<L>;
	unsigned long long inputSize = sizeof(BitSequence<K>)* N, outputSize = sizeof(BitSequence<L>);
	timerMemory.Start();
	CHECK_ERRORS(cudaMalloc(&d_idata, inputSize));
	CHECK_ERRORS(cudaMalloc(&d_odata, outputSize));
	CHECK_ERRORS(cudaMemcpy(d_idata, h_sequence, inputSize, cudaMemcpyHostToDevice));
	CHECK_ERRORS(cudaMemcpy(d_odata, h_odata, outputSize, cudaMemcpyHostToDevice));
	timerCall.Start();
	unsigned long long offset = 0;
#ifdef ITER_GPU
	for (; offset + B * 1024 < L; offset += B * 1024)
	{
		checkSequencesGPU << < B, 1024 >> > (d_idata, d_odata, offset);
		CHECK_ERRORS(cudaDeviceSynchronize());
	}
	if (L - offset >= 1024)
	{
		checkSequencesGPU << < (int)((L - offset) / 1024), 1024 >> > (d_idata, d_odata, offset);
		offset += (L - offset) * 1024;
		CHECK_ERRORS(cudaDeviceSynchronize());
	}
	if ((L - offset) % 1024)
	{
		checkSequencesGPU << < 1, (int)(L - offset) >> > (d_idata, d_odata, offset);
		offset += L - offset;
		CHECK_ERRORS(cudaDeviceSynchronize());
	}
	CHECK_ERRORS(cudaDeviceSynchronize());
#else
	if (L >= 1024)
	{
		checkSequencesGPU <<< (int)(L / 1024), 1024 >>> (d_idata, d_odata, 0);
		CHECK_ERRORS(cudaDeviceSynchronize());
	}
	if (L % 1024)
	{
		checkSequencesGPU <<< 1, L % 1024 >>> (d_idata, d_odata, (L / 1024) * 1024);
		CHECK_ERRORS(cudaDeviceSynchronize());
	}
#endif
	xtime = timerCall.Stop();
	CHECK_ERRORS(cudaMemcpy(h_odata, d_odata, outputSize, cudaMemcpyDeviceToHost));
	xmtime = timerMemory.Stop();
	cudaFree(d_idata);
	cudaFree(d_odata);
	printf("GPU Times : execution: %f, with memory: %f\n", xtime, xmtime);
	//auto res = vector<pair<int, int>>();
	auto res = ToPairVector(*h_odata);
	delete h_odata;
	return res;
}

vector<pair<int, int> > findPairsCPU(BitSequence<K> * sequence)
{
	BitSequence<L> *odata;
	odata = new BitSequence<L>();
	CudaTimer timerCall;
	timerCall.Start();
	checkSequencesCPU(sequence, odata);
	float xtime = timerCall.Stop();
	printf("CPU execution time: %f\n", xtime);
	auto res = ToPairVector(*odata);
	delete odata;
	return res;
}
