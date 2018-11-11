#include "BitSerie.h"
#include <cuda_runtime_api.h>

template<unsigned int k>
BitSequence<k>::BitSequence()
{
}

template<unsigned int k>
BitSequence<k>::BitSequence(char array[])
{
	cudaMemcpy(this->array, array, arSize, cudaMemcpyHostToHost);
}

template<unsigned int k>
__host__ __device__ inline char BitSequence<k>::GetBit(unsigned int index)
{
	return array[index / 8] >> (index % 8) & 1;
}

template<unsigned int k>
__host__ __device__ inline void BitSequence<k>::SetBit(unsigned int index, char value)
{
	array[index / 8] = (array[index / 8] & (1 << (index % 8)) | (value << (index % 8)));
}

template<unsigned int k>
__host__ __device__ inline unsigned int *BitSequence<k>::GetWord32(unsigned int word_index)
{
	return (uint32_t*)(array+word_index*32/8);
}

template<unsigned int k>
__host__ __device__ inline unsigned long long *BitSequence<k>::GetWord64(unsigned int word_index)
{
	return (uint64_t*)(array+word_index*64/8);
}
