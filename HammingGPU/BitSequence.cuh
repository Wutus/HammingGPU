#pragma once
#include <cuda_runtime_api.h>

template<unsigned int k>
class BitSequence
{
public:
	__host__ __device__ BitSequence();
	__host__ __device__ BitSequence(char array[]);
	__host__ __device__ inline char GetBit(unsigned int index);
	__host__ __device__ inline void SetBit(unsigned int index, char value);
	__host__ __device__ inline unsigned int *GetWord32(unsigned int word_index);
	__host__ __device__ inline unsigned long long *GetWord64(unsigned int word_index);
	static const unsigned int arSize = (k + 7) / 8;
private:
	char array[arSize];
};

