#pragma once
#include <cuda_runtime_api.h>

template<unsigned int k>
class BitSequence
{
public:
	__host__ BitSequence()
	{ 
	}
	__host__ BitSequence(char array[])
	{
		cudaMemcpy(this->array, array, arSize, cudaMemcpyHostToHost);
	}
	__host__ __device__ inline char GetBit(unsigned int index)
	{
		return array[index / 8] >> (index % 8) & 1;
	}
	__host__ __device__ inline void SetBit(unsigned int index, char value)
	{
		array[index / 8] = (array[index / 8] & (~(1 << (index % 8)))) | ((!!value) << (index % 8));
	}
	__host__ __device__ inline unsigned int *GetWord32(unsigned int word_index)
	{
		return (unsigned int*)(array + word_index * 32 / 8);
	}
	__host__ __device__ inline unsigned long long *GetWord64(unsigned int word_index)
	{
		return (unsigned long long*)(array + word_index * 64 / 8);
	}
	static const unsigned int arSize = (k + 7) / 8;
private:
	char array[arSize];
};

/*void f()
{
	BitSequence<1000> bs;
	BitSequence<1000000> bs2;
	bs.GetBit(0);
	bs.SetBit(0, 0);
	bs.GetWord32(0);
	bs.GetWord64(0);
	bs2.GetBit(0);
	bs2.GetWord32(0);
	bs2.GetWord64(0);
	bs2.SetBit(0, 0);
}*/