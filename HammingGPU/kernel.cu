#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

#define ITER_GPU //To omit watchdog on windows
//#define LINUX

#define CHECK_ERRORS(status) do{\
	if(cudaSuccess != status) {\
		fprintf(stderr, "Cuda Error in %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(status));\
	}\
}while(0)

#define CHECK_ERRORS_FORMAT(status, format, ...) do{\
	if(cudaSuccess != status) {\
		fprintf(stderr, "Cuda Error in %s:%d - %s - ", __FILE__, __LINE__, cudaGetErrorString(status));\
		fprintf(stderr, format, __VA_ARGS__);\
		fprintf(stderr, "\n");\
	}\
}while(0)

#define BITS_IN_SEQUENCE 10000 //Number of bits in one sequence
#define INPUT_SEQUENCE_SIZE 100000ull //Number of sequences
#define COMPARISONS (((INPUT_SEQUENCE_SIZE*(INPUT_SEQUENCE_SIZE - 1)) / 2)) //Number of comparisons
#define MAX_BLOCKS 100000 //Number of maximum blocks per call
#define THREADS_PER_BLOCK 1024 //Number of threads run in one block
#define SEQUENCES_PER_CALL 35 //Number of rows taken once per function call

template<unsigned long long k>
class BitSequence;
class CudaTimer;

template<unsigned int N>
class DeviceResultArray;

class ResultArray
{
public:
	unsigned int **arr;
};

template<unsigned int N>
class HostResultArray : public ResultArray
{
public:

	HostResultArray()
	{
		arr = new unsigned int*[N - 1];

		for (int i = 0; i < N - 1; i++)
		{
			arr[i] = new unsigned int[(int)(ceil((i + 1) / 32.0))];
		}
	}

	~HostResultArray()
	{
		if (arr == nullptr)
			return;

		for (int i = 0; i < N - 1; i++)
		{
			delete[] arr[i];
		}

		delete[] arr;
	}

	HostResultArray<N>&& operator=(HostResultArray<N> &&h_result)
	{
		this->arr = h_result.arr;
		h_result.arr = nullptr;
	}

	HostResultArray(HostResultArray<N> &&h_result)
	{
		this->arr = h_result.arr;
		h_result.arr = nullptr;
	}

	void CopyRows(const DeviceResultArray<N> & array, unsigned int start, unsigned int quantity)
	{
		unsigned int **temp_arr = new unsigned int*[quantity];
		cudaMemcpy(temp_arr, array.arr + start - 1, quantity * sizeof(unsigned int*), cudaMemcpyDeviceToHost);
		for (int i = 0; i < quantity; ++i)
		{
			cudaMemcpyAsync(arr[start - 1 + i], temp_arr[i], sizeof(unsigned int) * (int)(ceil((start + i) / 32.0)), cudaMemcpyDeviceToHost);
		}
		delete[] temp_arr;
	}

	char GetBit(unsigned int row, unsigned int col) const
	{
		return (char)(arr[row - 1][col / 32] >> (col % 32) & 1);
	}
};

template<unsigned int N>
class DeviceResultArray : public ResultArray
{
public:
	DeviceResultArray()
	{
		CHECK_ERRORS(cudaMalloc(&arr, sizeof(unsigned int*)*(N - 1)));
		unsigned int* temp_arr[N - 1];
		for (int i = 0; i < N - 1; ++i)
		{
			CHECK_ERRORS(cudaMalloc(&(temp_arr[i]), sizeof(unsigned int) * (ceil((i + 1) / 32.0))));
			CHECK_ERRORS(cudaMemset(temp_arr[i], 0, sizeof(unsigned int) * (ceil((i + 1) / 32.0))));
		}
		CHECK_ERRORS(cudaMemcpyAsync(arr, &(temp_arr[0]), sizeof(unsigned int*)*(N - 1), cudaMemcpyHostToDevice));
		CHECK_ERRORS(cudaDeviceSynchronize());
	}

	~DeviceResultArray()
	{
		unsigned int *temp_arr[N - 1];
		CHECK_ERRORS(cudaMemcpy(temp_arr, arr, sizeof(unsigned int*)*(N - 1), cudaMemcpyDeviceToHost));
		for (int i = 0; i < N - 1; i++)
		{
			CHECK_ERRORS(cudaFree(temp_arr[i]));
		}
		CHECK_ERRORS(cudaFree(arr));
	}

	HostResultArray<N> ToHostArray()
	{
		HostResultArray<N> host;
		unsigned int * temp_arr[N - 1];
		CHECK_ERRORS(cudaMemcpy(temp_arr, arr, sizeof(unsigned int*)*(N - 1), cudaMemcpyDeviceToHost));
		for (int i = 0; i < N - 1; ++i)
		{
			CHECK_ERRORS(cudaMemcpyAsync(host.arr[i], temp_arr[i], sizeof(unsigned int) * (unsigned int)(ceil((i + 1) / 32.0)), cudaMemcpyDeviceToHost));
		}
		CHECK_ERRORS(cudaDeviceSynchronize());
		return host;
	}
};

__global__ void Hamming1GPU(BitSequence<BITS_IN_SEQUENCE> * d_sequence, BitSequence<COMPARISONS> *d_odata, unsigned long long offset = 0);
__host__ __device__ char compareSequences(BitSequence<BITS_IN_SEQUENCE> * sequence1, BitSequence<BITS_IN_SEQUENCE> * sequence2);
__host__ __device__ void k2ij(unsigned long long k, unsigned int * i, unsigned int  * j);
__host__ __device__ unsigned long long ij2k(unsigned int i, unsigned int j);
void Hamming1CPU(BitSequence<BITS_IN_SEQUENCE> * sequence, BitSequence<COMPARISONS> * odata);
void PrintComparison(const BitSequence<BITS_IN_SEQUENCE> & gpu_sequence, const BitSequence<BITS_IN_SEQUENCE> & cpu_sequence);
bool ComparePairs(const vector<pair<int, int> > & gpu_result, const vector<pair<int, int> > & cpu_result);

ostream & operator<<(ostream & out, BitSequence<BITS_IN_SEQUENCE> & sequence);
BitSequence<BITS_IN_SEQUENCE> * GenerateInput();
vector<pair<int, int> > ToPairVector(const BitSequence<COMPARISONS> & result_sequence);
void PrintAsMatrix(const BitSequence<COMPARISONS> & sequence, ostream & stream);

vector<pair<int, int> > FindPairsGPU(BitSequence<BITS_IN_SEQUENCE> * h_sequence);
vector<pair<int, int> > FindPairsGPU2(BitSequence<BITS_IN_SEQUENCE> * h_sequence);
vector<pair<int, int> > FindPairsCPU(BitSequence<BITS_IN_SEQUENCE> * sequence);
__host__ __device__ unsigned int* GetPointer(unsigned int **arr, unsigned int row, unsigned int col);
template<unsigned int N>
vector<pair<int, int> > ToPairVector(const HostResultArray<N> & result_array);

void PrintArray(BitSequence<BITS_IN_SEQUENCE> * arr);

int main()
{
	cudaError_t cudaStatus;
	printf("Starting sequence generation...\n");
	BitSequence<BITS_IN_SEQUENCE>* sequence = GenerateInput();
	printf("Ended sequence generation!\n");

	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	auto gpuRes = FindPairsGPU2(sequence);
	cout << "On first call found " << gpuRes.size() << " results" << endl;
	cout.flush();
	auto gpuRes0 = FindPairsGPU(sequence);
	cout << "On second call found " << gpuRes0.size() << " results" << endl;
	cout.flush();
	auto cpuRes = FindPairsCPU(sequence);
	cout << "On third call found " << cpuRes.size() << " results" << endl;
	//PrintArray(sequence);
	//ComparePairs(gpuRes, cpuRes);
	//ComparePairs(gpuRes0, cpuRes);
	/*if (equal(gpuRes.begin(), gpuRes.end(), gpuRes0.begin(), gpuRes0.end()))
	{
		cout << "Result from new GPU implementation are OK" << endl;
	}*/

	//if (equal(gpuRes0.begin(), gpuRes0.end(), cpuRes.begin()))
	//{
		//cout << "Result from old GPU implementation are OK" << endl;
	//}
	//ComparePairs(gpuRes0, cpuRes);

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
	__host__ __device__ BitSequence()
	{

	}
	__host__ __device__ inline char GetBit(unsigned long long index) const
	{
		return (array[index / 64] >> (index % 64)) & 1;
	}
	__host__ __device__ inline void SetBit(unsigned long long index, char value)
	{
		array[index / 64] = (array[index / 64] & (~(1ull << (index % 64)))) | ((!!value) << (index % 64));
	}
	__host__ __device__ inline unsigned int *GetWord32(unsigned int word_index)
	{
		return (((unsigned int*)array) + word_index);
	}
	__host__ __device__ inline unsigned long long *GetWord64(unsigned int word_index)
	{
		return (array + word_index);
	}

	__host__ __device__ BitSequence(const BitSequence<k> & sequence)
	{
		memcpy(array, sequence.array, arSize * 8);
	}

	__host__ __device__ const BitSequence<k> & operator=(const BitSequence<k> & sequence)
	{
		memcpy(array, sequence.array, arSize * 8);
		return sequence;
	}
	static const unsigned long long arSize = (k + 63) / 64;
private:
	unsigned long long array[arSize];
};

__host__ __device__ char compareSequences(BitSequence<BITS_IN_SEQUENCE> * sequence1, BitSequence<BITS_IN_SEQUENCE> * sequence2)
{
	int diff = 0;
	for (int j = 0; j < (BITS_IN_SEQUENCE + 63) / 64; ++j)
	{
		unsigned long long int a, b, axorb;
		a = *(sequence1->GetWord64(j));
		b = *(sequence2->GetWord64(j));
		axorb = a ^ b;
		diff += axorb == 0 ? 0 : (axorb & (axorb - 1) ? 2 : 1);
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
	*j = (unsigned int)((k + 1) - 0.5 * (*i) * ((unsigned long long)(*i) - 1)) - 1;
}

__host__ __device__ unsigned long long ij2k(unsigned int i, unsigned int j)
{
	return ((unsigned long long)i) * (i - 1) / 2 + j;
}

void Hamming1CPU(BitSequence<BITS_IN_SEQUENCE> * sequence, BitSequence<COMPARISONS> * odata)
{
	unsigned long long numberOfComparisons = COMPARISONS;
	int i1 = 1, i2 = 0;
	for (unsigned long long k = 0; k < numberOfComparisons / 32; ++k)
	{
		unsigned int result = 0;
		for (int i = 0; i < 32; i++)
		{
			result |= (unsigned int)(compareSequences(sequence + i1, sequence + i2)) << i;
			++i2;
			if (i2 == i1)
			{
				++i1;
				i2 = 0;
			}
		}
		*(odata->GetWord32(k)) = result;
	}
	if (numberOfComparisons % 32)
	{
		unsigned int result = 0;
		for (int i = 0; i < numberOfComparisons % 32; i++)
		{
			result |= (unsigned int)(compareSequences(sequence + i1, sequence + i2)) << i;
			++i2;
			if (i2 == i1)
			{
				++i1;
				i2 = 0;
			}
		}
		*(odata->GetWord32(numberOfComparisons / 32)) = result;
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

void PrintComparison(const BitSequence<BITS_IN_SEQUENCE> & gpu_sequence, const BitSequence<BITS_IN_SEQUENCE> & cpu_sequence)
{
	for (unsigned long long i = 0; i < INPUT_SEQUENCE_SIZE*(INPUT_SEQUENCE_SIZE - 1) / 2; ++i)
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

	vector<pair<int, int> > gpu_res(gpu_result);
	vector<pair<int, int> > cpu_res(cpu_result);
	sort(gpu_res.begin(), gpu_res.end());
	sort(cpu_res.begin(), cpu_res.end());

	const vector<pair<int, int> > & gv = csize > gsize ? cpu_res : gpu_res;
	bool equal = true;

	if (gsize != csize)
	{
		cout << "Number of elements is not equal (GPU: " << gsize << ", CPU: " << csize << ") !" << endl;
		equal = false;
	}
	else
	{
		cout << "Number of elements are equal (GPU: " << gsize << ", CPU: " << csize << ")" << endl;
	}

	int i;
	for (i = 0; i < n; ++i)
	{
		if (gpu_res[i] != cpu_res[i])
		{
			cout << "Difference on " << i << ": GPU: (" << gpu_res[i].first << ", " << gpu_res[i].second << ") CPU: ("
				<< cpu_res[i].first << ", " << cpu_res[i].second << ")" << endl;
			equal = false;
		}
		else
		{
			//cout << "Pair " << i << ": GPU: (" << gpu_res[i].first << ", " << gpu_res[i].second << ") CPU: ("
			//		<< cpu_res[i].first << ", " << cpu_res[i].second << ")" << endl;

		}

	}
	if (csize != gsize)
	{
		cout << "Rest pairs on " << ((csize > gsize) ? "CPU" : "GPU") << ":" << endl;
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

__global__ void Hamming1GPU(BitSequence<BITS_IN_SEQUENCE> * d_sequence, BitSequence<COMPARISONS> *d_odata, unsigned long long offset)
{
	unsigned long long i = threadIdx.x + blockIdx.x * blockDim.x + offset;
	unsigned int i1, i2;
	unsigned int c, b;
	k2ij(i, &i1, &i2);
	c = compareSequences(d_sequence + i1, d_sequence + i2);
#ifdef LINUX
	__syncthreads();
	b = __ballot(c);
#else
	b = __ballot_sync(~0, c);
#endif
	if(!!(i % 32))
		*(d_odata->GetWord32(i / 32)) = b;
}

ostream & operator<<(ostream & out, BitSequence<BITS_IN_SEQUENCE> & sequence)
{
	for (unsigned long long i = 0; i < BITS_IN_SEQUENCE; ++i)
	{
		out << (short int)sequence.GetBit(i);
	}
	return out;
}

BitSequence<BITS_IN_SEQUENCE> * GenerateInput()
{
	srand(2018);

	BitSequence<BITS_IN_SEQUENCE> * r = new BitSequence<BITS_IN_SEQUENCE>[INPUT_SEQUENCE_SIZE];

	memset(r, 0, sizeof(BitSequence<BITS_IN_SEQUENCE>)*INPUT_SEQUENCE_SIZE);

	for (int i = 0; i < INPUT_SEQUENCE_SIZE; i++)
	{
		*(r[i].GetWord32(0)) = i;
		/*for (int j = 0; j < BITS_IN_SEQUENCE / 32; j++)
		{
			*(r[i].GetWord32(j)) = rand() + rand()*RAND_MAX;
		}
		if(BITS_IN_SEQUENCE % 32)
			*(r[i].GetWord32(BITS_IN_SEQUENCE / 32)) = (rand() + rand()*RAND_MAX)%(1<<(BITS_IN_SEQUENCE%32));*/
	}
	return r;
}

vector<pair<int, int> > ToPairVector(const BitSequence<COMPARISONS> & result_sequence)
{
	vector<pair<int, int> > result;
	for (unsigned long long k = 0; k < COMPARISONS; k++)
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

void PrintAsMatrix(const BitSequence<COMPARISONS> & sequence, ostream & stream)
{
	for (int i = 0; i < INPUT_SEQUENCE_SIZE; ++i)
	{
		for (int j = 0; j < INPUT_SEQUENCE_SIZE; ++j)
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

vector<pair<int, int> > FindPairsGPU(BitSequence<BITS_IN_SEQUENCE> * h_sequence)
{
	BitSequence<BITS_IN_SEQUENCE> *d_idata;
	BitSequence<COMPARISONS> *h_odata, *d_odata;
	CudaTimer timerCall, timerMemory;
	float xtime, xmtime;
	h_odata = new BitSequence<COMPARISONS>;
	unsigned long long inputSize = sizeof(BitSequence<BITS_IN_SEQUENCE>)* INPUT_SEQUENCE_SIZE, outputSize = sizeof(BitSequence<COMPARISONS>);
	timerMemory.Start();
	CHECK_ERRORS(cudaMalloc(&d_idata, inputSize));
	CHECK_ERRORS(cudaMalloc(&d_odata, outputSize));
	CHECK_ERRORS(cudaMemcpy(d_idata, h_sequence, inputSize, cudaMemcpyHostToDevice));
	CHECK_ERRORS(cudaMemcpy(d_odata, h_odata, outputSize, cudaMemcpyHostToDevice));
	timerCall.Start();
#ifdef ITER_GPU
	unsigned long long offset = 0;
	for (; offset + MAX_BLOCKS * THREADS_PER_BLOCK < COMPARISONS; offset += MAX_BLOCKS * (unsigned long long) THREADS_PER_BLOCK)
	{
		Hamming1GPU << < MAX_BLOCKS, THREADS_PER_BLOCK >> > (d_idata, d_odata, offset);
		CHECK_ERRORS_FORMAT(cudaDeviceSynchronize(), "offset=%llu", offset);
	}
	if (COMPARISONS - offset >= THREADS_PER_BLOCK)
	{
		Hamming1GPU << < (int)((COMPARISONS - offset) / THREADS_PER_BLOCK), THREADS_PER_BLOCK >> > (d_idata, d_odata, offset);
		offset += (COMPARISONS - offset) / THREADS_PER_BLOCK * THREADS_PER_BLOCK;
		//CHECK_ERRORS(cudaDeviceSynchronize());
	}
	if ((COMPARISONS - offset) % THREADS_PER_BLOCK)
	{
		Hamming1GPU << < 1, (int)(COMPARISONS - offset) >> > (d_idata, d_odata, offset);
		offset += COMPARISONS - offset;
		//CHECK_ERRORS(cudaDeviceSynchronize());
	}
	CHECK_ERRORS(cudaDeviceSynchronize());
#else
	if (COMPARISONS >= THREADS_IN_BLOCK)
	{
		Hamming1GPU << < (int)(COMPARISONS / THREADS_IN_BLOCK), THREADS_IN_BLOCK >> > (d_idata, d_odata, 0);
		CHECK_ERRORS(cudaDeviceSynchronize());
	}
	if (COMPARISONS % THREADS_IN_BLOCK)
	{
		Hamming1GPU << < 1, COMPARISONS % THREADS_IN_BLOCK >> > (d_idata, d_odata, (COMPARISONS / THREADS_IN_BLOCK) * THREADS_IN_BLOCK);
		CHECK_ERRORS(cudaDeviceSynchronize());
	}
#endif
	xtime = timerCall.Stop();
	CHECK_ERRORS(cudaMemcpy(h_odata, d_odata, outputSize, cudaMemcpyDeviceToHost));
	xmtime = timerMemory.Stop();
	cudaFree(d_idata);
	cudaFree(d_odata);
	printf("GPU Times : execution: %f, with copying memory: %f\n", xtime, xmtime);

	vector<pair<int, int> > res = ToPairVector(*h_odata);
	delete h_odata;
	return res;
}

vector<pair<int, int> > FindPairsCPU(BitSequence<BITS_IN_SEQUENCE> * sequence)
{
	BitSequence<COMPARISONS> *odata;
	odata = new BitSequence<COMPARISONS>();
	CudaTimer timerCall;
	timerCall.Start();
	Hamming1CPU(sequence, odata);
	float xtime = timerCall.Stop();
	printf("CPU execution time: %f\n", xtime);
	vector<pair<int, int> > res = ToPairVector(*odata);
	delete odata;
	return res;
}

__host__ __device__ inline char CompareWords64(const unsigned long long & first, const unsigned long long & second)
{
	unsigned long long x = first ^ second;

	//printf("x = %llu, f = %llu, s = %llu, res = %d, x&(x-1) = %llu\n", x, first, second, (int)(x > 0? ((x & (x - 1)) ? 2 : 1) : 0), (x & (x - 1)));
	return (x ? ((x & (x - 1)) ? 2 : 1) : 0);
}

#define WORDS64_IN_SEQUENCE ((BITS_IN_SEQUENCE + 63) / 64)

__global__ void Hamming2GPUFast(BitSequence<BITS_IN_SEQUENCE> *sequences, unsigned int **arr, unsigned int row_offset, unsigned int column_offset)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int seq_no = tid + column_offset;
	char res[SEQUENCES_PER_CALL];
	memset(res, 0, SEQUENCES_PER_CALL * sizeof(char));

	BitSequence<BITS_IN_SEQUENCE> & s = *(sequences + seq_no);
	__shared__ BitSequence<BITS_IN_SEQUENCE> ar[SEQUENCES_PER_CALL];
	for (unsigned int offset = 0; offset < SEQUENCES_PER_CALL * WORDS64_IN_SEQUENCE; offset += blockDim.x)
	{
		unsigned int oid = threadIdx.x + offset;
		if (oid < SEQUENCES_PER_CALL * WORDS64_IN_SEQUENCE)
		{
			*(ar[oid / WORDS64_IN_SEQUENCE].GetWord64(oid % WORDS64_IN_SEQUENCE)) =
				*((sequences + row_offset - oid / WORDS64_IN_SEQUENCE)->GetWord64(oid % WORDS64_IN_SEQUENCE));
			//printf("Thread %d wrote %llu\n", oid, *(ar[oid / WORDS64_IN_SEQUENCE].GetWord64(oid % WORDS64_IN_SEQUENCE)));
		}
	}
	__syncthreads();
	for (int j = 0; j < WORDS64_IN_SEQUENCE; ++j)
	{
		unsigned long long sf = *(s.GetWord64(j));
		for (int i = 0; i < SEQUENCES_PER_CALL; ++i)
		{
			//unsigned int seq2_no = row_offset - i;
			if (res[i] <= 1)
			{
				char r = CompareWords64(sf, *(ar[i].GetWord64(j)));
				res[i] += r;
			}
		}
	}
	for (int i = 0; i < SEQUENCES_PER_CALL; ++i)
	{
		unsigned int b;
		unsigned int seq2_no = row_offset - i;
		char v = res[i] == 1;
#ifdef LINUX
		__syncthreads();
		b = __ballot(v);
#else
		b = __ballot_sync(~0, v);
#endif
		if(!(seq_no % 32))
			*(GetPointer(arr, seq2_no, seq_no)) = b;
	}
}

__global__ void Hamming2GPU(BitSequence<BITS_IN_SEQUENCE> *sequences, unsigned int **arr, unsigned int row_offset, unsigned int column_offset)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int seq_no = tid + column_offset;
	char res[SEQUENCES_PER_CALL];
	char m = SEQUENCES_PER_CALL > row_offset ? row_offset : SEQUENCES_PER_CALL;
	memset(res, 0, SEQUENCES_PER_CALL * sizeof(char));

	BitSequence<BITS_IN_SEQUENCE> & s = *(sequences + seq_no);
	__shared__ BitSequence<BITS_IN_SEQUENCE> ar[SEQUENCES_PER_CALL];
	for (unsigned int offset = 0; offset < m * WORDS64_IN_SEQUENCE; offset += blockDim.x)
	{
		unsigned int oid = threadIdx.x + offset;
		if (oid < m * WORDS64_IN_SEQUENCE)
		{
			*(ar[oid / WORDS64_IN_SEQUENCE].GetWord64(oid % WORDS64_IN_SEQUENCE)) =
				*((sequences + row_offset - oid / WORDS64_IN_SEQUENCE)->GetWord64(oid % WORDS64_IN_SEQUENCE));
			//printf("Thread %d wrote %llu\n", oid, *(ar[oid / WORDS64_IN_SEQUENCE].GetWord64(oid % WORDS64_IN_SEQUENCE)));
		}
	}
	__syncthreads();
	for (int j = 0; j < WORDS64_IN_SEQUENCE; ++j)
	{
		unsigned long long sf = *(s.GetWord64(j));
		for (int i = 0; i < m; ++i)
		{
			//unsigned int seq2_no = row_offset - i;
			if (res[i] <= 1)
			{
				unsigned long long s = *(ar[i].GetWord64(j));
				char r = CompareWords64(sf, s);
				//printf("%d %d %d %d % d\n", seq2_no, seq_no, j, i, r);
				res[i] += r;
			}
		}
	}
	for (int i = 0; i < m; ++i)
	{
		unsigned int b;
		unsigned int seq2_no = row_offset - i;
		char v = res[i] == 1;
#ifdef LINUX
		__syncthreads();
		b = __ballot(v);
#else
		b = __ballot_sync(~0, v);
#endif
		//printf("%d %d %d %d %d\n", seq_no, seq2_no, (int)v, b, (int)res[i]);
		if (seq2_no > seq_no && !(seq_no%32))
		{
			*(GetPointer(arr, seq2_no, seq_no)) = b;
		}
	}
}

vector<pair<int, int> > FindPairsGPU2(BitSequence<BITS_IN_SEQUENCE> * h_sequence)
{
	BitSequence<BITS_IN_SEQUENCE> *d_idata;
	DeviceResultArray<INPUT_SEQUENCE_SIZE> d_result;
	CudaTimer timerCall, timerMemory;
	float xtime, xmtime;
	unsigned long long inputSize = sizeof(BitSequence<BITS_IN_SEQUENCE>)* INPUT_SEQUENCE_SIZE;
	timerMemory.Start();
	//HostResultArray<(unsigned int)INPUT_SEQUENCE_SIZE> h_result;
	CHECK_ERRORS(cudaMalloc(&d_idata, inputSize));
	CHECK_ERRORS(cudaMemcpy(d_idata, h_sequence, inputSize, cudaMemcpyHostToDevice));
	timerCall.Start();
	unsigned int copied = INPUT_SEQUENCE_SIZE;
	for (int j = INPUT_SEQUENCE_SIZE - 1; j > 0; j -= SEQUENCES_PER_CALL)
	{
		if (j >= THREADS_PER_BLOCK)
		{
			Hamming2GPUFast <<< j / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>> (d_idata, d_result.arr, j, 0);
			//CHECK_ERRORS(cudaDeviceSynchronize());
		}
		if (j % THREADS_PER_BLOCK > 0)
		{
			Hamming2GPU <<< 1, j%THREADS_PER_BLOCK >>> (d_idata, d_result.arr, j, j - (j%THREADS_PER_BLOCK));
			//CHECK_ERRORS_FORMAT(cudaDeviceSynchronize(), "%d %d", j, (j%THREADS_PER_BLOCK));
		}
		/*if (j < INPUT_SEQUENCE_SIZE / 2 && copied == INPUT_SEQUENCE_SIZE)
		{
			CHECK_ERRORS(cudaDeviceSynchronize());
			unsigned int start = j - SEQUENCES_PER_CALL > 0 ? j - SEQUENCES_PER_CALL + 1 : 1;
			unsigned int quantity = SEQUENCES_PER_CALL > j ? j : SEQUENCES_PER_CALL;
			//cout << j << " " << start << " " << quantity << endl;
			h_result.CopyRows(d_result, start, copied - j);
			copied = j;
		}*/
	}
	//h_result.CopyRows(d_result, 1, copied - 1);
	CHECK_ERRORS(cudaDeviceSynchronize());
	xtime = timerCall.Stop();
	HostResultArray<INPUT_SEQUENCE_SIZE> h_result(d_result.ToHostArray());
	xmtime = timerMemory.Stop();
	cudaFree(d_idata);
	printf("GPU Times : execution: %f, with memory: %f\n", xtime, xmtime);
	//vector<pair<int,int> > res = vector<pair<int, int>>();
	vector<pair<int, int> > res = ToPairVector(h_result);
	return res;
}

template<unsigned int N>
vector<pair<int, int> > ToPairVector(const HostResultArray<N> & result_array)
{
	vector<pair<int, int> > result;
	for (int i = 1; i < N; ++i)
	{
		for (int j = 0; j < i; ++j)
		{
			if (result_array.GetBit(i, j))
			{
				result.push_back(make_pair(i, j));
			}
		}
	}

	return result;
}

__host__ __device__ unsigned int* GetPointer(unsigned int **arr, unsigned int row, unsigned int col)
{
	return arr[row - 1] + col / 32;
}

void PrintArray(BitSequence<BITS_IN_SEQUENCE> * arr)
{
	for (int i = 0; i < INPUT_SEQUENCE_SIZE; ++i)
	{
		cout << arr[i] << endl;
	}
}