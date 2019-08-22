#include <stdio.h>
#include <stdlib.h>
#include "mat.h"
#include "cnn.h"

int main()
{
	int ii = 0, jj = 0;
	int out;
	nSize mat_size9 = { 9,9 };
	nSize mat_size8 = { 8,8 };
	nSize mat_size7 = { 7,7 };
	nSize mat_size6 = { 6,6 };
	nSize mat_size5 = { 5,5 };
	nSize mat_size4 = { 4,4 };
	nSize mat_size3 = { 3,3 };
	nSize mat_size2 = { 2,2 };
	nSize inputSize = { 128,128 };
	//nSize inSize = mat_size5;
	//nSize mapSize = mat_size3;
	//nSize outSize;

	float*** inData = (float***)malloc(sizeof(float**));
	//float*** outData = (float***)malloc(sizeof(float**));
	//float**** mapData = (float****)malloc(sizeof(float***));
	//float* biasData = (float*)malloc(sizeof(float));

	CNN* cnn=(CNN*)malloc(sizeof(CNN));

	*inData = create_mat(inputSize);
	// init indata
	for (ii = 0; ii < inputSize.c; ii++)
		for (jj = 0; jj < inputSize.r; jj++)
		{
			*(*(*inData + ii) + jj) = (float)(ii+1.0)*(jj+1.0);
		}

	cnnsetup(cnn, inputSize, 9);
	cnnload(cnn, "my_model.csv");
	out = cnnff(cnn,inData);
	printf("%d\n", out);

	while (1);
	return 0;
}

/*
	*outData = create_mat(mat_size5);
	*mapData = (float***)malloc(sizeof(float**));
	**mapData = create_mat(mapSize);
	*biasData = 0;

	//printmat(*inData, inSize);
	for (ii = 0; ii < mapSize.c; ii++)
		for (jj = 0; jj < mapSize.r; jj++)
		{
			*(*(**mapData + ii) + jj) = 1.0 -2 * (jj / 2);
		}

//CovLayer* initCovLayer(int inputHeight, int inputWidth, int mapSize, int strides, int inChannels, int outChannels, int PaddingType)
CovLayer* C1 = initCovLayer(inSize.r, inSize.c, mapSize.c, 1, 1, 1, same);
//PoolLayer* initPoolLayer( int inputHeight, int inputWidth, int mapSize,  int strides, int Channels, int paddingtype,int pooltype)
PoolLayer* P2 = initPoolLayer(C1->outputHeight, C1->outputWidth, 2, 2, C1->outChannels, same, AvgPool);
// global pool
PoolLayer* P3 = initPoolLayer(P2->outputHeight, P2->outputWidth, P2->outputHeight, 1, P2->Channels, valid, AvgPool);
covlloaddir(C1, mapData, biasData);
//printmat(**(C1->mapData), matSize);
//printf("%d\n", *(C1->biasData));
covlff(C1, inData);
//poollff(P2, C1->y);
poollff(P2, inData);
outSize.r = P2->outputHeight;
outSize.c = P2->outputWidth;
printmat(*(P2->y), outSize);
//printf("%d", P3->outputHeight);
poollff(P3, P2->y);
outSize.r = P3->outputHeight;
outSize.c = P3->outputWidth;
printmat(*(P3->y), outSize);
//printf("%d\n",C1->PaddingType);

printf("end");
*/