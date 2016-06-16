#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "time.h"
#include <helper_math.h>


#define BEAM 8
#define CBITS 12

unsigned char *TransSeq;
unsigned char *TransChar;
unsigned char *TransCRC;
unsigned int *TransInt;
unsigned int *TransState;
unsigned long *TransRan;
float *TransRNGBin;
unsigned char *RecvChar;
unsigned int *RecvState;
unsigned long *RecvRan;
unsigned int PktLen;
unsigned int TransLen = 8;
unsigned int CharLen = TransLen / 8;
unsigned int CRCLen;
unsigned int IntLen;
unsigned int RNGBinLen;
unsigned int PktNum;
unsigned int *state;
unsigned int EncodePass;
unsigned int HalfLen;
unsigned int TotalLen;
unsigned int FinalCharLen;
unsigned int c_bits;
unsigned int EncodePassSussess = 1;
unsigned long rndseqseed;
unsigned int *ThisTimeTransSeq;
unsigned int BeamWidth;
unsigned int BeamLen;

float a0, a1, b0, b1, a11, a00;

unsigned int *mi0;
unsigned int *mi;
unsigned int *si0;
unsigned int *si;
unsigned int *RN0;
unsigned int *RN;
float *cost0;
float *cost;
unsigned int **record;
float *costtemp;
unsigned *statetemp;
unsigned int *index;
unsigned int *indextemp;
unsigned int *finalresult;
unsigned char *crcresult;

unsigned int *mi0Gpu;
unsigned int *si0Gpu;
unsigned int *miGpu;
unsigned int *siGpu;
unsigned int *stateGpu;
unsigned long *mtGpu;
int *mtiGpu;
unsigned long *mag01Cpu;
unsigned long *mag01Gpu;
unsigned int *RN0Gpu;
unsigned int *RNGpu;
float *TransRNGBinCpu;
float *TransRNGBinGpu;
float *costGpu;

__device__ float b0_Gpu;
__device__ float b1_Gpu;
__device__ unsigned int TotalLen_Gpu;
__device__ unsigned int c_bits_Gpu;
unsigned int *EncodePass_Gpu;
unsigned int *EncodePass_Cpu;

#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL /* least significant r bits */
static unsigned long mt[N]; /* the array for the state vector  */
static int mti=N+1; /* mti==N+1 means mt[N] is not initialized */

#define PI 3.14159265358979

__global__ void InitGenrandKernel(unsigned int *ss, unsigned long *mt, int *mti)
{
	int myID = threadIdx.x + blockIdx.x * blockDim.x;

	mti[myID] = N + 1;
	unsigned long s = ss[myID];
	
    mt[myID * N + 0]= s & 0xffffffffUL;
    for (mti[myID] = 1; mti[myID] < N; mti[myID]++) 
	{
        mt[myID * N + mti[myID]] = 
	    (1812433253UL * (mt[myID * N + mti[myID] - 1] ^ (mt[myID * N + mti[myID] - 1] >> 30)) + mti[myID]); 
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
        mt[myID * N + mti[myID]] &= 0xffffffffUL;
        /* for >32 bit machines */
    }

    __syncthreads();
}

__global__ void GenrandInt32Kernel(unsigned int *yrand, unsigned long *mt, int *mti, unsigned long *mag01)
{
	int myID = threadIdx.x + blockIdx.x * blockDim.x;
	
    unsigned long y;
    
    //static unsigned long mag01[2] = {0x0UL, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti[myID] >= N) 
    { /* generate N words at one time */
        int kk;

        //if (mti[myID] == N+1)   /* if init_genrand() has not been called, */
            //init_genrand(5489UL); /* a default initial seed is used */

        for (kk = 0; kk < N-M; kk++) 
		{
            y = (mt[myID * N + kk] & UPPER_MASK) | (mt[myID * N + kk + 1] & LOWER_MASK);
            mt[myID * N + kk] = mt[myID * N + kk + M] ^ (y >> 1) ^ mag01[myID * 2 + y & 0x1UL];
        }
        for (; kk < N-1; kk++) 
		{
            y = (mt[myID * N + kk] & UPPER_MASK) | (mt[myID * N + kk + 1] & LOWER_MASK);
            mt[myID * N + kk] = mt[myID * N + kk + (M - N)] ^ (y >> 1) ^ mag01[myID * 2 + y & 0x1UL];
        }
        y = (mt[myID * N + N - 1] & UPPER_MASK) | (mt[myID * N + 0] & LOWER_MASK);
        mt[myID * N + N-1] = mt[myID * N + M - 1] ^ (y >> 1) ^ mag01[myID * 2 + y & 0x1UL];

        mti[myID] = 0;
    }
  
    y = mt[myID * N + (mti[myID]++)];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    yrand[myID] = ((unsigned int)y >> (32 - c_bits_Gpu));

    __syncthreads();
}

__global__ void HashFuncKernel(unsigned int *spine_value, unsigned int *message_block, unsigned int *state)
{
	int myID = threadIdx.x + blockIdx.x * blockDim.x;  
	
	state[myID * 3 + 0] = 0xdeadbeef + spine_value[myID];
	state[myID * 3 + 1] = 0xdeadbeef + spine_value[myID];
	state[myID * 3 + 2] = 0xdeadbeef + spine_value[myID];

	state[myID * 3 + 1] += message_block[myID];
	state[myID * 3 + 2] ^= state[myID * 3 + 1];
	state[myID * 3 + 2] -= ((state[myID * 3 + 1] << 14)|(state[myID * 3 + 1] >> (32 - 14)));
	state[myID * 3 + 0] ^= state[myID * 3 + 2];
	state[myID * 3 + 0] -= ((state[myID * 3 + 2] << 11)|(state[myID * 3 + 2] >> (32 - 11)));
	state[myID * 3 + 1] ^= state[myID * 3 + 0];
	state[myID * 3 + 1] -= ((state[myID * 3 + 0] << 25)|(state[myID * 3 + 0] >> (32 - 25)));
	state[myID * 3 + 2] ^= state[myID * 3 + 1];
	state[myID * 3 + 2] -= ((state[myID * 3 + 1] << 16)|(state[myID * 3 + 1] >> (32 - 16)));
	state[myID * 3 + 0] ^= state[myID * 3 + 2];
	state[myID * 3 + 0] -= ((state[myID * 3 + 2] << 4)|(state[myID * 3 + 2] >> (32 - 4)));
	state[myID * 3 + 1] ^= state[myID * 3 + 0];
	state[myID * 3 + 1] -= ((state[myID * 3 + 0] << 14)|(state[myID * 3 + 0] >> (32 - 14)));
	state[myID * 3 + 2] ^= state[myID * 3 + 1];
	state[myID * 3 + 2] -= ((state[myID * 3 + 1] << 24)|(state[myID * 3 + 1] >> (32 - 24)));

	spine_value[myID] = state[myID * 3 + 2];

	__syncthreads();
}

__global__ void CostComputeKernel(unsigned int *RN, float *RecvRNBin, float *cost)
{
	int myID = threadIdx.x + blockIdx.x * blockDim.x;

	int i;
	unsigned int indexRN;
	float costcomputed = 0;
	
	for(i = 0; i < c_bits_Gpu; i++)
	{
		indexRN = RN[myID];
		indexRN = ((indexRN & (0x1 << (c_bits_Gpu - (i + 1)))) >> (c_bits_Gpu - (i + 1)));
		if(1 == indexRN)
		{
			costcomputed = costcomputed + abs(b1_Gpu - RecvRNBin[myID + TotalLen_Gpu * i]);
		}
		else
		{
			costcomputed = costcomputed + abs(b0_Gpu - RecvRNBin[myID + TotalLen_Gpu * i]);
		}
	}

	cost[myID] = costcomputed + cost[myID];

	__syncthreads();
}

__global__ void V2CBitsRNKernel(unsigned int *RN)
{
	int myID = threadIdx.x + blockIdx.x * blockDim.x;
	
	unsigned int ramdonnumber;

	ramdonnumber = RN[myID];
	RN[myID] = ramdonnumber >> (32 - c_bits_Gpu);
}

void OOK_init(float SNR)
{
	float e = (float)1.6 * 1e-19;;
	float Is = (float)2 * 1e-9;
	float Ts = (float)5 * 1e-10;
	float f = (float)0.7 / Ts;
	float k = (float)1.38 * 1e-23;
	float T = (float)300;
	float B = (float)PI / 2 * f;
	float Cp = (float)300 * 1e-15;
	float Rf = (float)1 / 2 / PI / f / Cp;
	float Pb = (float)3 * 1e-14;
	float R0 = (float)0.875;
	float A = (float)pow(10, SNR/10);
	float a = (float)0.10;

	float Ps_temp = (e * e + (2 * e * Is + 4 * k * T * B / Rf + Pb * Pb * R0 * R0) * (1 / A - a * a));
	float Ps = (e + pow(Ps_temp, float(0.5))) / (R0 * (1 / A - a * a));

	float m = Is * Ts / e;
	float sita2 = 2 * B * Ts * (Is * Ts / e + 2 * k * T * Ts / Rf / e / e);
	float n0 = R0 / e * Ts * (Pb + a * Ps);
	float n1 = R0 / e * Ts * (Pb + Ps);

	a1 = n1 + sita2;
	a0 = n0 + sita2;
	b1 = n1 + m;
	b0 = n0 + m;

	a11 = pow(a1, float(0.5));
	a00 = pow(a0, float(0.5));

	printf("a1: %f\n", a1);
	printf("a0: %f\n", a0);
	printf("b1: %f\n", b1);
	printf("b0: %f\n", b0);
}


void init_genrand(unsigned long s)
{
    mt[0]= s & 0xffffffffUL;
    for (mti=1; mti<N; mti++) 
	{
        mt[mti] = 
	    (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti); 
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
        mt[mti] &= 0xffffffffUL;
        /* for >32 bit machines */
    }
}

unsigned long genrand_int32(void)
{
    unsigned long y;
    static unsigned long mag01[2]={0x0UL, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= N) { /* generate N words at one time */
        int kk;

        if (mti == N+1)   /* if init_genrand() has not been called, */
            init_genrand(5489UL); /* a default initial seed is used */

        for (kk=0;kk<N-M;kk++) 
		{
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (;kk<N-1;kk++) 
		{
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
        mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

        mti = 0;
    }
  
    y = mt[mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    //return (y >> (32 - c_bits));
    return y;
}

float Gaussrand()
{
    static float U, V;
    static int phase = 0;
    float Z;

    if(phase == 0)
    {
         U = (genrand_int32() + 1.) / (0xffffffff + 2.);
         V = genrand_int32() / (0xffffffff + 1.);
         Z = sqrt(-2 * log(U)) * sin(2 * PI * V);
    }
    else
    {
         Z = sqrt(-2 * log(U)) * cos(2 * PI * V);
    }

    phase = 1 - phase;

    return Z;
}

void PrintSeqChar(unsigned char *Seq, unsigned int seqlen)
{
	unsigned int i;
	
	for(i = 0; i < seqlen; i++)
	{
		printf("0x%x ", Seq[i]);
	}
	printf("\n");

	return;
}

void PrintSeqInt(unsigned int *Seq, unsigned int seqlen)
{
	unsigned int i;
	
	for(i = 0; i < seqlen; i++)
	{
		printf("0x%x ", Seq[i]);
	}
	printf("\n");

	return;
}

void PrintSeqLong(unsigned long *Seq, unsigned int seqlen)
{
	unsigned int i;
	
	for(i = 0; i < seqlen; i++)
	{
		printf("0x%lx ", Seq[i]);
	}
	printf("\n");

	return;
}

void PrintSeqFloat(float *Seq, unsigned int seqlen)
{
	unsigned int i;
	
	for(i = 0; i < seqlen; i++)
	{
		printf("%f ", Seq[i]);
	}
	printf("\n");

	return;
}


void ReadSeq(unsigned int seqlen)
{
	FILE *fp;
	unsigned int i;

	//printf("ReadSeq.\n");
	
	fp = fopen("X.txt", "r");
	for(i = 0; i < seqlen; i++)
	{
		fscanf(fp, "%d ", &TransSeq[i]);
	}

	fclose(fp);
	fp = NULL;

	return;
}

void Bin2Hex(unsigned int passnum)
{
	unsigned int i, j;
	float mean = 1;
	float variance = 1;
	float rnd;

	if(1 == passnum)
	{
		init_genrand(rndseqseed);
		for(i = 0; i < TransLen; i++)
		{
			rnd = mean + variance * (float)Gaussrand();
			//printf("%f\n", rnd);
			if(mean < rnd)
			{
				TransSeq[i] = 1;
			}
			else
			{
				TransSeq[i] = 0;
			}
			ThisTimeTransSeq[i] = TransSeq[i];
		}
	}
	else
	{
		for(i = 0; i < TransLen; i++)
		{
			TransSeq[i] = ThisTimeTransSeq[i];
		}
	}
	
	for(i = 0; i < CharLen; i++)
	{
		for(j = 0; j < 8; j++)
		{
			TransChar[i] = TransChar[i] + (TransSeq[i * 8 + j] << (8 - (j + 1)));
		}
	}

	return;
}

void Hex2Bin()
{
	unsigned int i, j;

	for(i = 0; i < IntLen; i++)
	{
		for(j = 0; j < c_bits; j++)
		{
			TransRNGBin[i * c_bits + j] = ((TransRan[i] & (0x1 << (c_bits - (j + 1)))) >> (c_bits - (j + 1)));
		}
	}
}

void Char2Int()
{
	unsigned int i, j, k;
	unsigned int num = 8 / PktLen;
	unsigned char temp;
	
	for(i = 0; i < CRCLen; i++)
	{
		for(j = 0; j < num; j++)
		{
			for(k = 0; k < PktLen; k++)
			{	
				temp = TransCRC[i];
				TransInt[i * num + j] = TransInt[i * num + j] + (temp & (0x1 << (8 - (j * PktLen + k + 1))));
			}
			temp = TransInt[i * num + j];
			TransInt[i * num + j] = temp >> ((num - (j + 1)) * PktLen);
		}
	}

	return;
}

void Int2Char()
{
	unsigned int i, j;
	unsigned int temp;

	for(i = 0; i < FinalCharLen; i++)
	{
		for(j = 0; j < (8 / PktLen); j++)
		{
			temp = finalresult[i * (8 / PktLen) + j] << (8 - (j + 1) * PktLen);
			crcresult[i] = crcresult[i] + temp;
		}
	}
}

void ChannelGaussian()
{
	unsigned int i;

	init_genrand(long(time(0)));

	for(i = 0; i < RNGBinLen; i++)
	{
		if(0 == TransRNGBin[i])
		{
			TransRNGBin[i] = b0 + a00 * (float)Gaussrand();
		}
		else
		{
			TransRNGBin[i] = b1 + a11 * (float)Gaussrand();
		}
		if(0 > TransRNGBin[i])TransRNGBin[i] = 0;
	}

	return;
}

unsigned short int CRCSeq(unsigned char *charseq, unsigned int len)
{
    unsigned short int crc = 0;
    unsigned int i;
   
    //printf("CRC Init.\n");
    //printf("CRC Calculating.\n");

	for(i = 0; i < len; i++)
	{
		TransCRC[i] = charseq[i];
	}

	crc = 0;
	//len = CharLen;
	unsigned char *ptr = charseq;
    while(len--) 
    {
        for(i = 0x80; i != 0; i = i >> 1) 
        {
         	if((crc & 0x8000) != 0)
         	{
            	crc = crc << 1;
            	crc = crc ^ 0x1021;
         	}
         	else 
         	{
            	crc = crc << 1;
        	}
        	if((*ptr & i) != 0)
        	{
          		crc = crc ^ 0x1021;
        	}
     	}
     	ptr++;
   	}
   	ptr = NULL;
   	//printf("0x%x \n", crc);
   	TransCRC[CharLen] = (crc & 0xff00) >> 8;
	TransCRC[CharLen + 1] = (crc & 0x00ff);
	//PrintSeqChar(TransCRC, len + 2);
	//printf("\n");

    //printf("CRC Finish.\n");
    //printf("---------------------\n");
   
    return crc;
}

unsigned int rot(unsigned int x, int k)
{
	return ((x << k)|(x >> (32 - k)));
}

unsigned int HashFunc(unsigned int spine_value, unsigned int message_block)
{
	state[0] = 0xdeadbeef + spine_value;
	state[1] = 0xdeadbeef + spine_value;
	state[2] = 0xdeadbeef + spine_value;

	state[1] += message_block;
	state[2] ^= state[1];
	state[2] -= rot(state[1],14);
	state[0] ^= state[2];
	state[0] -= rot(state[2],11);
	state[1] ^= state[0];
	state[1] -= rot(state[0],25);
	state[2] ^= state[1];
	state[2] -= rot(state[1],16);
	state[0] ^= state[2];
	state[0] -= rot(state[2],4);
	state[1] ^= state[0];
	state[1] -= rot(state[0],14);
	state[2] ^= state[1];
	state[2] -= rot(state[1],24);

	return state[2];
}

void HashSeq()
{
	unsigned int i;

	//printf("ENCODEPASS: %d\n", EncodePass);
	TransState[0] = HashFunc(EncodePass, TransInt[0]);
	for(i = 0; i < IntLen - 1; i++)
	{
		TransState[i + 1] = HashFunc(TransState[i], TransInt[i + 1]);
	}

	return;
}

void RNG()
{
	unsigned int i;

	for(i = 0; i < IntLen; i++)
	{
		//init_genrand((unsigned long)TransState[i]);
		//TransRan[i] = genrand_int32();
		TransRan[i] = ((HashFunc(TransState[i], EncodePass)) >> (32 - c_bits));
	}

	return;
}

void PrepareM()
{
	unsigned int i, j;

	for(i = 0; i < HalfLen; i++)
	{
		mi0[i] = i;
	}
	for(i = 0; i < TotalLen; i++)
	{
		mi[i] = (i % HalfLen);
	}
	for(i = 0; i < RNGBinLen; i++)
	{
		for(j = 0; j < TotalLen; j++)
		{
			TransRNGBinCpu[i * TotalLen + j] = TransRNGBin[i];
		}
	}
	//PrintSeqFloat(TransRNGBinCpu, RNGBinLen * TotalLen);
	cudaMemcpy(miGpu, mi, sizeof(unsigned int) * TotalLen, cudaMemcpyHostToDevice);
	cudaMemcpy(TransRNGBinGpu, TransRNGBinCpu, sizeof(float) * RNGBinLen * TotalLen, cudaMemcpyHostToDevice);
	cudaMemcpy(costGpu, cost, sizeof(float) * TotalLen, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(b0_Gpu, &b0, sizeof(float));
	cudaMemcpyToSymbol(b1_Gpu, &b1, sizeof(float));
	cudaMemcpyToSymbol(TotalLen_Gpu, &TotalLen, sizeof(unsigned int));
	cudaMemcpyToSymbol(c_bits_Gpu, &c_bits, sizeof(unsigned int));
	for(i = 0; i < TotalLen; i++)
	{
		EncodePass_Cpu[i] = EncodePass;
	}
	cudaMemcpy(EncodePass_Gpu, EncodePass_Cpu, sizeof(unsigned int) * TotalLen, cudaMemcpyHostToDevice);

	return;
}

void BubbleSort(unsigned int len, unsigned int k)
{
	unsigned int i, j;
	unsigned int tempint;
	float tempfloat;
	unsigned int tempstate;
	unsigned int offset;
	
	//printf("BubbleSort.\n");
	
	if(HalfLen == len)
	{
		//indextemp[i] = i + (indextemp[i] << (PktLen * 2));
		for(i = 0; i < len; i++)
		{
			for(j = 0; j < (len - 1); j++)
			{
				if(cost0[j] > cost0[j + 1])
				{
					tempfloat = cost0[j];
					tempint = index[j];
					tempstate = si0[j];
					cost0[j] = cost0[j + 1];
					index[j] = index[j + 1];
					si0[j] = si0[j + 1];
					cost0[j + 1] = tempfloat;
					index[j + 1] = tempint;
					si0[j + 1] = tempstate;
				}
			}
		}
		for(i = 0; i < BeamLen; i++)
		{
			record[k][i] = index[i];
			for(j = 0; j < HalfLen; j++)
			{
				cost[i * HalfLen + j] = cost0[i];
				si[i * HalfLen + j] = si0[i];
			}
		}
		//PrintSeqInt(index, TotalLen);
	}
	else
	{
		//PrintSeqInt(index, TotalLen);
		for(i = 0; i < BeamLen; i++)
		{
			//Keep the safty of the length of indexes, so Pktlen is adopted. 
			offset = PktLen * 2;
			for(j = 0; j < HalfLen; j++)
			{
				indextemp[i * HalfLen + j] = ((i * HalfLen + j) + (index[i] << offset)) & (0xffffffff >> (32 - PktLen * 4));
				//indextemp[i * HalfLen + j] = (j + (index[i] << offset)) & (0xffffffff >> (32 - PktLen * 4));
				//index[i * HalfLen + j] = indextemp[i * HalfLen + j];
				//PrintSeqInt(index + i, 1);
			}
		}
		//PrintSeqInt(indextemp, TotalLen);
		//printf("-----------------------\n");
		for(i = 0; i < len; i++)
		{
			index[i] = indextemp[i];
		}
		//PrintSeqInt(index, TotalLen);
		for(i = 0; i < len; i++)
		{
			//printf("BubbleSort -- ROW: %d\n", i);
			for(j = 0; j < (len - 1); j++)
			{
				if(cost[j] > cost[j + 1])
				{
					tempfloat = cost[j];
					tempint = index[j];
					tempstate = si[j];
					cost[j] = cost[j + 1];
					index[j] = index[j + 1];
					si[j] = si[j + 1];
					cost[j + 1] = tempfloat;
					index[j + 1] = tempint;
					si[j + 1] = tempstate;
				}
			}
		}
		for(i = 0; i < BeamLen; i++)
		{
			record[k][i] = index[i];
		}
		for(i = 0; i < len; i++)
		{
			costtemp[i] = cost[i];
			statetemp[i] = si[i]; 
		}
		//PrintSeqInt(index, TotalLen);
		for(i = 0; i < BeamLen; i++)
		{
			for(j = 0; j < HalfLen; j++)
			{
				cost[i * HalfLen + j] = costtemp[i];
				si[i * HalfLen + j] = statetemp[i];
			}
		}
	}

	return;
}

unsigned int CheckCRCResult()
{
	unsigned int i, j;
	unsigned int temp1, temp2;
	unsigned short int crc;
	unsigned int crcstate = 1;

	finalresult[IntLen - 1] = record[IntLen - 1][0] & ((0xffffffff >> (32 - PktLen * 4)) >> (PktLen * 2));
	temp1 = (record[IntLen - 1][0] & (0xffffffff >> (32 - PktLen * 4))) >> (PktLen * 2);
	for(i = 1; i < IntLen; i++)
	{
		for(j = 0; j < BeamLen; j++)
		{
			temp2 = record[IntLen - 1 - i][j] & ((0xffffffff >> (32 - PktLen * 4)) >> (PktLen * 2));
			//printf("temp2 = 0x%x\n", temp2);
			if(temp2 == temp1)
			{
				finalresult[IntLen - 1 - i] = temp2;
				//printf("F = 0x%x\n", finalresult[IntLen - 1 - i]);
				temp1 = (record[IntLen - 1 - i][j] & (0xffffffff >> (32 - PktLen * 4))) >> (PktLen * 2);
				//printf("temp1 = 0x%x\n", temp1);
				j = BeamLen;
			}
		}
	}
	//PrintSeqInt(finalresult, IntLen);
	Int2Char();
	
	crc = CRCSeq(crcresult, FinalCharLen);
	if(!crc)
	{
		//printf("Decoding Complete.\n");
		crcstate = 0;
	}

	return crcstate;
}

void ClearTrans()
{
	unsigned int i, j;
	
	for(i = 0; i < TransLen; i++)
	{
		TransSeq[i] = 0;
	}
	for(i = 0; i < CharLen; i++)
	{
		TransChar[i] = 0;
	}
	for(i = 0; i < CRCLen; i++)
	{
		TransCRC[i] = 0;
	}
	for(i = 0; i < IntLen; i++)
	{
		TransInt[i] = 0;
		TransState[i] = 0;
		TransRan[i] = 0;
		finalresult[i] = 0;
	}
	for(i = 0; i < RNGBinLen; i++)
	{
		TransRNGBin[i] = 0;
	}
	for(i= 0; i < HalfLen; i++)
	{
		mi0[i] = 0;
		si0[i] = EncodePass;
		RN0[i] = 0;
		cost0[i] = 0;
	}
	for(i = 0; i < TotalLen; i++)
	{
		mi[i] = 0;
		si[i] = EncodePass;
		RN[i] = 0;
		cost[i] = 0;
		index[i] = i;
		indextemp[i] = i;
	}
	for(i = 0; i < PktNum; i++)
	{
		for(j = 0; j < HalfLen; j++)
		{
			record[i][j] = 0;
		}
	}
	for(i = 0; i < (TotalLen / 2); i++)
	{
		//mag01[2] = {0x0UL, MATRIX_A};
		mag01Cpu[i * 2] = 0x0UL;
		mag01Cpu[i * 2 + 1] = MATRIX_A;
	}
	for(i = 0; i < FinalCharLen; i++)
	{
		crcresult[i] = 0;
	}
	cudaMemcpy(mag01Gpu, mag01Cpu, sizeof(unsigned long) * TotalLen, cudaMemcpyHostToDevice);

	return;
}

void Init()
{	
	unsigned int i;
	
	printf("Init.\n");
	TransSeq = (unsigned char*)malloc(sizeof(unsigned char) * TransLen);
	TransChar = (unsigned char*)malloc(sizeof(unsigned char) * CharLen);
	TransCRC = (unsigned char*)malloc(sizeof(unsigned char) * CRCLen);
	TransInt = (unsigned int*)malloc(sizeof(unsigned int) * IntLen);
	state = (unsigned int*)malloc(sizeof(unsigned int) * 3);
	TransState = (unsigned int*)malloc(sizeof(unsigned int) * IntLen);
	TransRan = (unsigned long*)malloc(sizeof(unsigned long) * IntLen);
	TransRNGBin = (float*)malloc(sizeof(float) * RNGBinLen);

	RecvChar = (unsigned char*)malloc(sizeof(unsigned char) * 8);
	RecvState = (unsigned int*)malloc(sizeof(unsigned int) * BEAM);
	RecvRan = (unsigned long*)malloc(sizeof(unsigned long) * 8);

	mi0 = (unsigned int*)malloc(sizeof(unsigned int) * HalfLen);
	mi = (unsigned int*)malloc(sizeof(unsigned int) * TotalLen);
	si0 = (unsigned int*)malloc(sizeof(unsigned int) * HalfLen);
	si = (unsigned int*)malloc(sizeof(unsigned int) * TotalLen);
	RN0 = (unsigned int*)malloc(sizeof(unsigned int) * HalfLen);
	RN = (unsigned int*)malloc(sizeof(unsigned int) * TotalLen);
	cost0 = (float*)malloc(sizeof(float) * HalfLen);
	cost = (float*)malloc(sizeof(float) * TotalLen);
	record = (unsigned int**)malloc(sizeof(unsigned int*) * IntLen);
  	for (i = 0; i < IntLen; i++)
	{
  		record[i] = (unsigned int*)malloc(sizeof(unsigned int) * BeamLen);
  	}
  	mag01Cpu = (unsigned long*)malloc(sizeof(unsigned long) * TotalLen * 2);
	TransRNGBinCpu = (float*)malloc(sizeof(float) * RNGBinLen * TotalLen);
	costtemp = (float*)malloc(sizeof(float) * TotalLen);
	index = (unsigned int*)malloc(sizeof(unsigned int) * TotalLen);
	indextemp = (unsigned int*)malloc(sizeof(unsigned int) * TotalLen);
	statetemp = (unsigned int*)malloc(sizeof(unsigned int) * TotalLen);
	finalresult = (unsigned int*)malloc(sizeof(unsigned int) * IntLen);
	crcresult = (unsigned char*)malloc(sizeof(unsigned char) * FinalCharLen);
	EncodePass_Cpu = (unsigned int*)malloc(sizeof(unsigned int) * TotalLen);
	ThisTimeTransSeq = (unsigned int*)malloc(sizeof(unsigned int) * TotalLen);
	
  	cudaMalloc((void**)&mi0Gpu, sizeof(unsigned int) * HalfLen);
  	cudaMalloc((void**)&si0Gpu, sizeof(unsigned int) * HalfLen);
  	cudaMalloc((void**)&stateGpu, sizeof(unsigned int) * TotalLen * 3);
  	cudaMalloc((void**)&mtGpu, sizeof(unsigned long) * TotalLen * N);
  	cudaMalloc((void**)&mtiGpu, sizeof(int) * TotalLen);
  	cudaMalloc((void**)&RN0Gpu, sizeof(unsigned int) * HalfLen);
  	cudaMalloc((void**)&mag01Gpu, sizeof(unsigned long) * TotalLen * 2);
  	cudaMalloc((void**)&TransRNGBinGpu, sizeof(float) * RNGBinLen * TotalLen);
  	cudaMalloc((void**)&costGpu, sizeof(float) * TotalLen);
  	cudaMalloc((void**)&miGpu, sizeof(unsigned int) * TotalLen);
  	cudaMalloc((void**)&siGpu, sizeof(unsigned int) * TotalLen);
  	cudaMalloc((void**)&RNGpu, sizeof(unsigned int) * TotalLen);
  	cudaMalloc((void**)&EncodePass_Gpu, sizeof(unsigned int) * TotalLen);

	ClearTrans();

	return;
}

void Exit()
{
	unsigned int i;
	
	free(TransSeq);
	TransSeq = NULL;
	free(TransChar);
	TransChar = NULL;
	free(TransCRC);
	TransCRC= NULL;
	free(TransInt);
	TransInt = NULL;
	free(state);
	state = NULL;
	free(TransState);
	TransState = NULL;
	free(TransRan);
	TransRan = NULL;
	free(TransRNGBin);
	TransRNGBin = NULL;
	free(RecvChar);
	RecvChar = NULL;
	free(RecvState);
	RecvState = NULL;
	free(RecvRan);
	RecvRan = NULL;
	free(mi0);
	mi0 = NULL;
	free(mi);
	mi = NULL;
	free(si0);
	si0 = NULL;
	free(si);
	si = NULL;
	free(RN0);
	RN0 = NULL;
	free(RN);
	RN = NULL;
	free(cost0);
	cost0 = NULL;
	free(cost);
	cost = NULL;
	for (i = 0; i < PktNum; i++)
	{
  		free(record[i]);
		record[i] = NULL;
  	}
	free(record);
	record = NULL;
	free(mag01Cpu);
	mag01Cpu = NULL;
	free(TransRNGBinCpu);
	TransRNGBinCpu = NULL;
	free(costtemp);
	costtemp = NULL;
	free(index);
	index = NULL;
	free(indextemp);
	indextemp = NULL;
	free(statetemp);
	statetemp = NULL;
	free(finalresult);
	finalresult = NULL;
	free(crcresult);
	crcresult = NULL;
	free(EncodePass_Cpu);
	EncodePass_Cpu = NULL;
	free(ThisTimeTransSeq);
	ThisTimeTransSeq = NULL;

	cudaFree(mi0Gpu);
	cudaFree(si0Gpu);
	cudaFree(stateGpu);
	cudaFree(mtGpu);
	cudaFree(mtiGpu);
	cudaFree(RN0Gpu);
	cudaFree(mag01Gpu);
	cudaFree(TransRNGBinGpu);
	cudaFree(costGpu);
	cudaFree(miGpu);
	cudaFree(siGpu);
	cudaFree(RNGpu);
	cudaFree(EncodePass_Gpu);

	return;
}

int main()
{
	unsigned int i, k;
	//unsigned short int crc;
	//FILE *frc;
	float SNR;
	printf("SNR: ");
	scanf("%f", &SNR);
	OOK_init(SNR);
	
	printf("PktLen: ");
	scanf("%d", &PktLen);
	printf("C-Bits: ");
	scanf("%d", &c_bits);
	printf("BeamWidth: ");
	scanf("%d", &BeamWidth);
	
	unsigned int maxpass;
	printf("Max-Pass: ");
	scanf("%d", &maxpass);
	PktNum = TransLen / PktLen;
	CRCLen = CharLen + 2;
	IntLen = CRCLen * 8 / PktLen;
	RNGBinLen = IntLen * c_bits;
	EncodePass = 1;
	HalfLen = (0x1 << (PktLen));
	TotalLen = ((0x1 << (PktLen)) << (BeamWidth));
	FinalCharLen = IntLen / (8 / PktLen);
	BeamLen = (0x1 << (BeamWidth));
	printf("IntLen: %d, HalfLen: %d, TotalLen: %d, BeamLen: %d.\n", IntLen, HalfLen, TotalLen, BeamLen);
	unsigned int framenum;
	printf("FrameNum: ");
	scanf("%d", &framenum);
	unsigned int passnum;
	double totalpass = 0; 
	double RATE = 0;
	unsigned int decodestate = 1;

	unsigned int blocksize, threadsize;
	if(1024 < TotalLen)
	{
		blocksize = TotalLen / 1024;
		threadsize = 1024;
	}
	else
	{
		blocksize = 1;
		threadsize = TotalLen;
	}

	Init();
	printf("--------------------------------\n");
	printf("Decoding...\n");
	
	for(i = 0; i < framenum; i++)
	{
		//printf("New Frame!\n");
		decodestate = 1;
		passnum = 1;
		EncodePass = EncodePassSussess;
		rndseqseed = i;
		ClearTrans();
		
		while(decodestate)
		{
			if(passnum > maxpass)
			{
				break;
			}
			ClearTrans();

			ReadSeq(TransLen);
			//PrintSeqChar(TransSeq, TransLen);
			Bin2Hex(passnum);
			//PrintSeqChar(TransChar, CharLen);
			CRCSeq(TransChar, CharLen);
			Char2Int();
			//PrintSeqInt(TransInt, IntLen);
			HashSeq();
			//PrintSeqInt(TransState, IntLen);
			RNG();
			//PrintSeqLong(TransRan, IntLen);
			Hex2Bin();
			//PrintSeqFloat(TransRNGBin, RNGBinLen);
			ChannelGaussian();
			//PrintSeqFloat(TransRNGBin, RNGBinLen);
			/*
			for(i = 0; i < 32; i++)
			{
				frc = fopen("TransRNG.txt", "a+");
				fprintf(frc, "%f ", TransRNGBin[32 * 2 + i]);
	    		fclose(frc);
				frc = NULL;
			}
			*/

			PrepareM();
			//printf("M0:\n");
			//PrintSeqInt(mi0, HalfLen);
			//printf("M:\n");
			//PrintSeqInt(mi, TotalLen);
			cudaMemcpy(mi0Gpu, mi0, sizeof(unsigned int) * HalfLen, cudaMemcpyHostToDevice);
			cudaMemcpy(si0Gpu, si0, sizeof(unsigned int) * HalfLen, cudaMemcpyHostToDevice);
			HashFuncKernel<<<1, HalfLen>>>(si0Gpu, mi0Gpu, stateGpu);
			cudaMemcpy(si0, si0Gpu, sizeof(unsigned int) * HalfLen, cudaMemcpyDeviceToHost);		
			//PrintSeqInt(si0, HalfLen);
			//InitGenrandKernel<<<1, HalfLen>>>(si0Gpu, mtGpu, mtiGpu);
			//GenrandInt32Kernel<<<1, HalfLen>>>(RN0Gpu, mtGpu, mtiGpu, mag01Gpu);
			cudaMemcpy(RN0Gpu, si0Gpu, sizeof(unsigned int) * HalfLen, cudaMemcpyDeviceToDevice);
			HashFuncKernel<<<1, HalfLen>>>(RN0Gpu, EncodePass_Gpu, stateGpu);
			V2CBitsRNKernel<<<1, HalfLen>>>(RN0Gpu);
			//cudaMemcpy(RN0, RN0Gpu, sizeof(unsigned int) * HalfLen, cudaMemcpyDeviceToHost);		
			//PrintSeqInt(RN0, HalfLen);
			//CostComputeKernel<<<1, HalfLen>>>(RN0Gpu, TransRNGBinGpu, costGpu);
			CostComputeKernel<<<1, HalfLen>>>(RN0Gpu, TransRNGBinGpu, costGpu);
			cudaMemcpy(cost0, costGpu, sizeof(float) * HalfLen, cudaMemcpyDeviceToHost);
			//PrintSeqFloat(cost0, HalfLen);
			BubbleSort(HalfLen, 0);
			//PrintSeqFloat(cost0, HalfLen);
			//PrintSeqInt(si0, HalfLen);
			//PrintSeqInt(si, TotalLen);

			for(k = 1; k < IntLen; k++)
			{
				//PrintSeqFloat(cost, TotalLen);
				/*
				for(i = 0; i < TotalLen; i++)
				{
					cost[i] = 0;
				}
				*/
				cudaMemcpy(siGpu, si, sizeof(unsigned int) * TotalLen, cudaMemcpyHostToDevice);
				cudaMemcpy(costGpu, cost, sizeof(float) * TotalLen, cudaMemcpyHostToDevice);
				//HashFuncKernel<<<1, TotalLen>>>(siGpu, miGpu, stateGpu);
				HashFuncKernel<<<blocksize, threadsize>>>(siGpu, miGpu, stateGpu);
				cudaMemcpy(si, siGpu, sizeof(unsigned int) * TotalLen, cudaMemcpyDeviceToHost);
				//PrintSeqInt(si, TotalLen);
				//InitGenrandKernel<<<1, TotalLen>>>(siGpu, mtGpu, mtiGpu);
				//GenrandInt32Kernel<<<1, TotalLen>>>(RNGpu, mtGpu, mtiGpu, mag01Gpu);
				//InitGenrandKernel<<<blocksize, threadsize>>>(siGpu, mtGpu, mtiGpu);
				//GenrandInt32Kernel<<<blocksize, threadsize>>>(RNGpu, mtGpu, mtiGpu, mag01Gpu);
				cudaMemcpy(RNGpu, siGpu, sizeof(unsigned int) * HalfLen, cudaMemcpyDeviceToDevice);
				HashFuncKernel<<<blocksize, threadsize>>>(RNGpu, EncodePass_Gpu, stateGpu);
				V2CBitsRNKernel<<<blocksize, threadsize>>>(RNGpu);
				//cudaMemcpy(RN, RNGpu, sizeof(unsigned int) * TotalLen, cudaMemcpyDeviceToHost);		
				//PrintSeqInt(RN, TotalLen);
				//CostComputeKernel<<<1, TotalLen>>>(RNGpu, TransRNGBinGpu + k * TotalLen * c_bits, costGpu);
				//CostComputeKernel<<<blocksize, threadsize>>>(RNGpu, TransRNGBinGpu + k * TotalLen * c_bits, costGpu);
				CostComputeKernel<<<blocksize, threadsize>>>(RNGpu, TransRNGBinGpu + k * TotalLen * c_bits, costGpu);
				cudaMemcpy(cost, costGpu, sizeof(float) * TotalLen, cudaMemcpyDeviceToHost);
				//PrintSeqFloat(cost, TotalLen);
				BubbleSort(TotalLen, k);
				//PrintSeqInt(si, TotalLen);
			}
			/*
			for(i = 0; i < IntLen; i++)
			{
				PrintSeqInt(record[i], HalfLen);
			}
			*/
			decodestate = CheckCRCResult();

			//printf("STATE: %d\n", decodestate);
			//printf("PASS: %d\n", passnum);
			if(decodestate)
			{
				passnum++;
				EncodePassSussess = 1;
				EncodePass++;
				//printf("Try a new pass\n");
			}
			else
			{
				totalpass = totalpass + (double)passnum;
				EncodePassSussess = EncodePass;
				//RATE = abs(RATE + TransLen * PktLen * 1e12 / ((double)(TransLen * passnum)) / c_bits / 1e12);
				RATE = abs(RATE + PktLen * 1e12 / ((double)(passnum)) / 1e12);
				printf("#%d -- Rate_Accumulated: %f, Pass_Num: %d.\n", i + 1, RATE, passnum);
			}
		}
		//printf("--------------------------------\n");
	}

	RATE = RATE / framenum;
	totalpass = abs(totalpass * 1e12 / (double)framenum / 1e12);

	printf("Complete.\n");
	printf("--------------------------------\n");
	printf("RATE: %f\n", RATE);
	printf("AVERAGE PASSES: %f\n", totalpass);

	Exit();
	cudaDeviceReset();

	return 0;
}
