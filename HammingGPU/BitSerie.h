#pragma once

template<unsigned int k>
class BitSequence
{
public:
	BitSequence();
	BitSequence(char array[]);
	inline char GetBit(unsigned int index);
	inline void SetBit(unsigned int index, char value);
	inline unsigned int *GetWord32(unsigned int word_index);
	inline unsigned long long *GetWord64(unsigned int word_index);
	static const unsigned int arSize = (k + 7) / 8;
private:
	char array[arSize];
};

