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

template<unsigned long long K>
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
template<unsigned long long N, unsigned long long K>
void checkSequencesCPU(BitSequence<K> * sequence, void * odatav)
{
	BitSequence<N*(N * 1) / 2> *odata = (BitSequence<N*(N * 1) / 2> *)odatav;
	unsigned long long numberOfComparisons = N * (N - 1) / 2;
	for (unsigned long long k = 0; k < numberOfComparisons; k += 32)
	{
		unsigned int result = 0;
		for (int i = 0; i < 32; i++)
		{
			unsigned int i1, i2;
			k2ij(k + i, &i1, &i2);
			result |= (unsigned int)(compareSequences<K>(sequence + i1, sequence + i2)) << i;
		}
		*(odata->GetWord32(k / 32)) = result;
	}
}

__host__ __device__ inline double sqrtp(unsigned long long a)
{
	return 0.0f;
}

__host__ __device__ inline void k2ij(unsigned long long  k, unsigned int * i, unsigned int  * j)
{
	//adding 1 to k to skip first result
	*i = (unsigned int)(0.5 * (-1 + sqrtl(1 + 8 * (k + 1))));
	//decreasing 1 from j , as we start from 0 not 1
	*j = (unsigned int)((k + 1) - 0.5 * (*i) * ((*i) - 1)) - 1;
}

__host__ __device__ inline unsigned long long ij2k(unsigned int i, unsigned int j)
{
	return i * (i - 1) / 2 + j;
}

template<unsigned long long N, unsigned long long K>
__global__ void checkSequencesGPU(BitSequence<K> * d_sequence, BitSequence<N*(N - 1) / 2> *d_odata, unsigned long long offset = 0)
{
	unsigned long long i = threadIdx.x + blockIdx.x * 512 + offset;
	unsigned int i1, i2;
	k2ij(i, &i1, &i2);
	i2 = compareSequences<K>(d_sequence + i1, d_sequence + i2);
	i1 = __ballot_sync(~0, compareSequences<K>(d_sequence + i1, d_sequence + i2));
	*(d_odata->GetWord32(i/32)) = i1;
}

template<unsigned long long N, unsigned long long K>
__global__ void checkSequencesGPU2(BitSequence<K> * d_sequence, BitSequence<N*(N - 1) / 2> *d_odata, unsigned long long offset = 0)
{
	//unsigned long long i = (unsigned long long)threadIdx.x + (unsigned long long)512 * (unsigned long long)blockIdx.x + offset;
	unsigned long long i = threadIdx.x + blockIdx.x * 512 + offset;
	unsigned int i1, i2;
	//printf("%d\n", blockIdx.x);
	k2ij(i, &i1, &i2);
	/*if (ij2k(i1, i2) != i)
	{
		printf("Error! ij2k not giving the same as k2ij! (i = %d, j = %d, k = %d)", i1, i2, i);
		return;
	}*/
	//*((unsigned int*)(d_odata + i / 32 * 4)) = i1; = 0;
	//((unsigned int*)(d_odata + i / 32 * 4)) = i1; = __ballot_sync(~0, compareSequences<K>(d_sequence + i1, d_sequence + i2));
	/*i2 = compareSequences<K>(d_sequence + i1, d_sequence + i2);
	if(!(i%32))
		*(unsigned int*)(d_odata + i / 32 * 4) = __ballot_sync(~0, i2);*/
		//*(unsigned int*)(d_odata + i / 32 * 4) = i1;
		//printf("Tid %d, i1 %d, i2 %d, res %d, bs %d\n", i, i1, i2, res, bs);
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
	}

	float Stop()
	{
		if (!started)
			return -1.0f;
		float ms;
		cudaEventRecord(stop);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms, start, stop);
		started = false;
		return ms;
	}
private:
	bool started;
	cudaEvent_t start, stop;
};

template<unsigned long long K>
void PrintComparison(const BitSequence<K> & gpu_sequence, const BitSequence<K> & cpu_sequence, unsigned long long N)
{
	for (unsigned long long i = 0; i < N*(N - 1) / 2; ++i)
	{
		if (cpu_sequence.GetBit(i) != gpu_sequence.GetBit(i))
		{
			unsigned int i1, i2;
			k2ij(i, &i1, &i2);
			cout << "Difference on comparison number " << i << " (" << i1 << ", " << i2 << ") GPU " << (short int)gpu_sequence.GetBit(i) << " CPU " << (short int)cpu_sequence.GetBit(i) << endl;
			unsigned int diff = 0;
			for (unsigned int j = 0; j < K; ++j)
			{
				diff += gpu_sequence[i1].GetBit(j) ^ cpu_sequence[i2].GetBit(j);
			}
			cout << "No of diffs: " << diff << endl;
		}
	}
}

bool ComparePairs(const vector<pair<int, int>> & gpu_result, const vector<pair<int, int>> & cpu_result)
{
	unsigned long long gsize = gpu_result.size(), csize = cpu_result.size();
	unsigned long long n = gsize < csize? gsize : csize;
	const vector<pair<int, int>> & lv = gsize < csize ? gpu_result : cpu_result;
	const vector<pair<int, int>> & gv = gsize < csize ? cpu_result : gpu_result;
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
	return equal;
}

const unsigned long long K = 10000; //Number of bits in one sequence
const unsigned long long N = 100000; //Number of sequences
const unsigned long long L = (N*(N - 1)) / 2; //Number of comparisons
const unsigned int B = 100; //Number of maximum blocks per call

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
	//memset(r, 0, sizeof(BitSequence<K>)*N);

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < K/32; j++)
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

void printAsMatrix(const vector<std::pair<int, int>>, ostream & stream)
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

vector<pair<int, int>> findPairsGPU(BitSequence<K> * h_sequence)
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
	/*for (; offset + B * 1024 < L; offset += B * 1024)
	{
		checkSequencesGPU<N, K> <<< B, 1024 >>> (d_idata, d_odata, offset);
		CHECK_ERRORS(cudaDeviceSynchronize());
	}
	if (L - offset >= 1024)
	{
		checkSequencesGPU<N, K> <<< (int)((L - offset) / 1024), 1024 >>> (d_idata, d_odata, offset);
		offset += (L - offset) * 1024;
		CHECK_ERRORS(cudaDeviceSynchronize());
	}
	if ((L - offset) % 1024)
	{
		checkSequencesGPU<N, K> <<< 1, (int)(L - offset) >>> (d_idata, d_odata, offset);
		offset += L - offset;
		CHECK_ERRORS(cudaDeviceSynchronize());
	}
	CHECK_ERRORS(cudaDeviceSynchronize());*/
	if (L >= 1024)
	{
		checkSequencesGPU<N, K> <<< (int)(L/1024), 1024 >> > (d_idata, d_odata, 0);
		CHECK_ERRORS(cudaDeviceSynchronize());
	}
	if (L % 1024)
	{
		checkSequencesGPU<N, K> <<< 1, L % 1024 >> > (d_idata, d_odata, (L/1024)*1024);
		CHECK_ERRORS(cudaDeviceSynchronize());
	}
	xtime = timerCall.Stop();
	CHECK_ERRORS(cudaMemcpy(h_odata, d_odata, outputSize, cudaMemcpyDeviceToHost));
	xmtime = timerMemory.Stop();
	cudaFree(d_idata);
	cudaFree(d_odata);
	printf("GPU Times : execution: %f, with memory: %f\n", xtime, xmtime);
	auto res = vector<pair<int, int>>();
	//auto res = ToPairVector(*h_odata);
	delete h_odata;
	return res;
}

vector<pair<int, int>> findPairsCPU(BitSequence<K> * sequence)
{
	BitSequence<L> *odata;
	odata = new BitSequence<L>();
	checkSequencesCPU<N,K>(sequence, odata);
	auto res = ToPairVector(*odata);
	delete odata;
	return res;
}

int main()
{
	cudaError_t cudaStatus;
	printf("Starting sequence generation...\n");
	BitSequence<K>* sequence = Generate();
	printf("Ended sequence generation!\n");

	auto gpuRes = findPairsGPU(sequence);
	//auto cpuRes = findPairsCPU(sequence);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

//vector<pair<int, int>> findPairs(BitSequence<K> * h_sequence)
//{
//	cudaError_t cudaStatus;
//	BitSequence<K> *d_sequence;
//
//	cudaStatus = cudaMalloc((void**)&d_sequence, sizeof(BitSequence<K>)*N);
//	if (cudaStatus != cudaSuccess)
//	{
//		fprintf(stderr, "cudaMalloc failed!  Do you have a CUDA-capable GPU installed?");
//	}
//
//	cudaStatus = cudaMemcpy(d_sequence, h_sequence, sizeof(BitSequence<K>)*N, cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess)
//	{
//		fprintf(stderr, "cudaMemcpy failed!  Do you have a CUDA-capable GPU installed?");
//	}
//	//Too big to keep on stack
//	BitSequence<L> *d_odata, *h_odata_p, *h_odata_p2;
//	h_odata_p = new BitSequence<L>;
//	h_odata_p2 = new BitSequence<L>;
//	BitSequence<L> & h_odata = *h_odata_p;
//	BitSequence<L> & h_odata2 = *h_odata_p2;
//	//printAsMatrix(h_odata, cout);
//	cudaStatus = cudaMalloc((void**)&d_odata, sizeof(BitSequence<L>));
//	if (cudaStatus != cudaSuccess)
//	{
//		fprintf(stderr, "cudaMallocfailed!  Do you have a CUDA-capable GPU installed?\n");
//	}
//	printf("Reserved %d under %d\n", sizeof(BitSequence<L>), d_odata);
//	cudaStatus = cudaMemcpy(h_odata_p, d_odata, sizeof(h_odata), cudaMemcpyDeviceToHost);
//	if (cudaStatus != cudaSuccess)
//	{
//		const char *err_str = cudaGetErrorString(cudaStatus);
//		fprintf(stderr, "cudaMemcpy no 1 failed! %s\n", err_str);
//	}
//	cudaEvent_t start, stop; 
//	float time;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//
//	cudaEventRecord( start, 0 );
//	printf("Starting counting on GPU...\n");
//	unsigned long long offset = 0;
//	unsigned long long nT = N * (N - 1) / 2;
//	/*for(unsigned long long i = 0; 1024*L*i < nT; ++i)
//	{
//		checkSequences<N, K> <<< L, 1024 >>> (d_sequence, d_odata, offset);
//		offset += L*1024;
//		printf("offset %ull\n", offset);
//		cudaStatus = cudaDeviceSynchronize();
//		if (cudaStatus != cudaSuccess)
//		{
//			const char *err_str = cudaGetErrorString(cudaStatus);
//			fprintf(stderr, "kernelCall failed on offset %ull! %s\n", err_str, offset);
//		}
//	}
//	checkSequences<N, K> <<<(nT%L), 1024 >>> (d_sequence, d_odata, offset);
//	offset += (nT%L)*1024;
//	printf("offset %ull\n", offset);
//	cudaStatus = cudaDeviceSynchronize();
//	if (cudaStatus != cudaSuccess)
//	{
//		const char *err_str = cudaGetErrorString(cudaStatus);
//		fprintf(stderr, "kernelCall failed on offset %ull! %s\n", err_str, offset);
//	}
//	checkSequences<N, K> <<<1, nT - offset >>> (d_sequence, d_odata, offset);
//	offset = nT;
//	printf("offset %ull\n", offset);*/
//	printf("Gonna run %llu blocks (%d), for %llu comparisons\n", L / 512, L / 512 < (1 << 30) - 1, L);
//	checkSequences<N, K> <<< (unsigned int)(L/512), 512 >>> (d_sequence, d_odata, 0);
//
//	/*cudaStatus = cudaGetLastError();
//	if (cudaStatus != cudaSuccess)
//	{
//		const char *err_str = cudaGetErrorString(cudaStatus);
//		fprintf(stderr, "cudaGetLastError failed! %s\n", err_str);
//	}*/
//
//	cudaStatus = cudaDeviceSynchronize();
//	if (cudaStatus != cudaSuccess)
//	{
//		const char *err_str = cudaGetErrorString(cudaStatus);
//		fprintf(stderr, "kernelCall failed (%s) on offset %llu!\n", err_str, offset);
//	}
//
//	offset = (nT / (unsigned long long)512) * (unsigned long long)512;
//	printf("offset: %llu\n", offset);
//	//checkSequences<N, K> <<<1, (unsigned int)(nT%512) >> > (d_sequence, d_odata, offset);
//
//	/*cudaStatus = cudaDeviceSynchronize();
//	if (cudaStatus != cudaSuccess)
//	{
//		const char *err_str = cudaGetErrorString(cudaStatus);
//		fprintf(stderr, "kernelCall failed on offset %ull! %s\n", err_str, offset);
//	}*/
//
//	cudaStatus = cudaGetLastError();
//	if (cudaStatus != cudaSuccess)
//	{
//		const char *err_str = cudaGetErrorString(cudaStatus);
//		fprintf(stderr, "cudaGetLastError failed! %s\n", err_str);
//	}
//
//	cudaEventRecord( stop, 0 );
//	cudaEventSynchronize( stop );
//
//	cudaEventElapsedTime( &time, start, stop );
//
//	cudaEventDestroy( start );
//	cudaEventDestroy( stop );
//    printf("GPU Processing time: %f (ms)\n", time);
//
//	cudaStatus = cudaGetLastError();
//	if (cudaStatus != cudaSuccess)
//	{
//		const char *err_str = cudaGetErrorString(cudaStatus);
//		fprintf(stderr, "cudaGetLastError failed! %s\n", err_str);
//	}
//
//	printf("sizeof(h_odata): %d, d_odata %d\n", sizeof(h_odata), sizeof(*d_odata));
//	cudaStatus = cudaMemcpy(h_odata_p, d_odata, sizeof(h_odata), cudaMemcpyDeviceToHost);
//	if (cudaStatus != cudaSuccess)
//	{
//		const char *err_str = cudaGetErrorString(cudaStatus);
//		fprintf(stderr, "cudaMemcpy failed! %s\n", err_str);
//	}
//	cudaStatus = cudaDeviceSynchronize();
//	if (cudaStatus != cudaSuccess)
//	{
//		const char *err_str = cudaGetErrorString(cudaStatus);
//		fprintf(stderr, "cudaDeviceSynchronize returned error code %s after launching addKernel!\nCurrent offset %llu", err_str, offset);
//	}
//	//printAsMatrix(h_odata, cout);
//
//	/*cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//	cudaEventRecord(start, 0);
//	cudaEventSynchronize(start);
//	printf("Starting counting on CPU... \n");
//	checkSequencesCPU<N, K>(h_sequence, &h_odata2);
//	
//	cudaEventRecord(stop, 0);
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&time, start, stop);
//
//	cudaEventDestroy(start);
//	cudaEventDestroy(stop);
//    printf("CPU Processing time: %f (ms)\n", time);
//	//printAsMatrix(h_odata2, cout);
//	cout << "Comparison: " << endl;
//	
//	//printAsMatrix(h_odata, cout);
//	
//	// ZWRACANIE TABLICY DWÓJEK*/
//	vector<pair<int, int>> result;
//
//	//cudaFree(d_sequence);
//	//cudaFree(d_odata);
//	//delete h_odata_p;
//	//delete h_odata_p2;
//	/*for (unsigned long long i = 0; i < N*(N-1)/2; ++i)
//	{
//		unsigned long long i1, i2;
//		k2ij(i, &i1, &i2);
//		cout << i1 << " " << i2 << ": " << (short int)h_odata.GetBit(i) << endl;
//	}*/
//	return result;
//}

/*BitSequence<L> *CPUHamming(BitSequence<K> * h_sequence)
{
	BitSequence<L> *h_odata_p = new BitSequence<L>;
	BitSequence<L> & h_odata2 = *h_odata_p;

	cudaEvent_t start, stop; float time;

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

	return h_odata_p;
}*/