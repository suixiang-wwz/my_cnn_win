#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "cnn.h"

void cnnsetup(CNN* cnn,nSize inputSize,int classes)
{
	CovP C1p = {5, 1, 8, same};
	PoolP P2p = {4, 3, same, MaxPool};
	CovP C3p = { 5, 1, 32, same };
	PoolP P4p = { 4, 3, same, MaxPool };
	PoolP G5p = { -1, 1, valid, AvgPool };
	CovP F6p = { 1, 1, classes, same };
	//CovLayer* initCovLayer(int inputHeight, int inputWidth, int inChannels, CovP* covp)
	//PoolLayer* initPoolLayer(int inputHeight, int inputWidth, int Channels, PoolP* poolp)
	cnn->C1 = initCovLayer(inputSize.r, inputSize.c, 1, &C1p);
	cnn->P2 = initPoolLayer(cnn->C1->outputHeight, cnn->C1->outputWidth,cnn->C1->outChannels,&P2p);
	cnn->C3 = initCovLayer(cnn->P2->outputHeight, cnn->P2->outputWidth, cnn->P2->Channels, &C3p);
	cnn->P4 = initPoolLayer(cnn->C3->outputHeight, cnn->C3->outputWidth, cnn->C3->outChannels, &P4p);
	G5p.mapSize = cnn->P4->outputHeight;
	cnn->G5 = initPoolLayer(cnn->P4->outputHeight, cnn->P4->outputWidth, cnn->P4->Channels, &G5p);
	cnn->F6 = initCovLayer(cnn->G5->outputHeight, cnn->G5->outputWidth, cnn->G5->Channels, &F6p);
}

// the forward propagation of conv layer
void covlff(CovLayer* covl, float*** inData)
{
	int i, j, r, c;
	nSize mapSize = { covl->mapSize,covl->mapSize };
	nSize inSize = { covl->inputWidth,covl->inputHeight };
	nSize outSize = { covl->outputWidth,covl->outputHeight };
	// reset v
	for (i = 0; i < (covl->outChannels); i++)
		for (r = 0; r < outSize.r; r++)
			for (c = 0; c < outSize.c; c++)
				covl->v[i][r][c] = 0.0;
	//printmat(*(covl->v), outSize);

	// forward propagation
	for (i = 0; i < (covl->outChannels); i++)
	{
		for (j = 0; j < (covl->inChannels); j++)
		{
			float** mapout = cov(covl->mapData[j][i], &mapSize, inData[j], &inSize, covl->strides, covl->PaddingType);
			//printmat(mapout, outSize);
			addmat(covl->v[i], covl->v[i], outSize, mapout, outSize);
			for (r = 0; r < outSize.r; r++)
				free(mapout[r]);
			free(mapout);
		}
		for (r = 0; r < outSize.r; r++)
			for (c = 0; c < outSize.c; c++)
				covl->y[i][r][c] = activation_reLU(covl->v[i][r][c], covl->biasData[i]);
	}
}
void covlff_noActi(CovLayer* covl, float*** inData)
{
	int i, j, r, c;
	nSize mapSize = { covl->mapSize,covl->mapSize };
	nSize inSize = { covl->inputWidth,covl->inputHeight };
	nSize outSize = { covl->outputWidth,covl->outputHeight };
	// reset v
	for (i = 0; i < (covl->outChannels); i++)
		for (r = 0; r < outSize.r; r++)
			for (c = 0; c < outSize.c; c++)
				covl->v[i][r][c] = 0.0;
	//printmat(*(covl->v), outSize);

	// forward propagation
	for (i = 0; i < (covl->outChannels); i++)
	{
		for (j = 0; j < (covl->inChannels); j++)
		{
			float** mapout = cov(covl->mapData[j][i], &mapSize, inData[j], &inSize, covl->strides, covl->PaddingType);
			//printmat(mapout, outSize);
			addmat(covl->v[i], covl->v[i], outSize, mapout, outSize);
			for (r = 0; r < outSize.r; r++)
				free(mapout[r]);
			free(mapout);
		}
		for (r = 0; r < outSize.r; r++)
			for (c = 0; c < outSize.c; c++)
				covl->y[i][r][c] = covl->v[i][r][c] + covl->biasData[i];
	}
}

void poollff(PoolLayer* pooll, float*** inData)
{
	if (pooll->PoolType == AvgPool)
		avgpoollff(pooll, inData);
	else if (pooll->PoolType == MaxPool)
		maxpoollff(pooll, inData);
	else
		printf("error pool type\n");
}
void maxpoollff(PoolLayer* pooll, float*** inData)
{
	int i,c,j,m,n;
	int strides = pooll->strides;
	int mapSize = pooll->mapSize;
	nSize inSize = { pooll->inputWidth,pooll->inputHeight };
	nSize outSize = { pooll->outputWidth,pooll->outputHeight };
	padSize psize;
	nSize exInSize;
	if (pooll->PaddingType == same)
	{
		// get padding size
		psize.r = (outSize.r - 1)*pooll->strides + pooll->mapSize - inSize.r;
		psize.c = (outSize.c - 1)*pooll->strides + pooll->mapSize - inSize.c;
		psize.r_top = psize.r / 2;
		psize.r_down = psize.r - psize.r_top;
		psize.c_top = psize.c / 2;
		psize.c_down = psize.c - psize.c_top;

		exInSize.r = psize.r + inSize.r;
		exInSize.c = psize.c + inSize.c;
		float** exInputData = (float**)malloc(exInSize.r * sizeof(float*));
		for (i = 0; i < exInSize.r; i++)
			exInputData[i] = (float*)calloc(exInSize.c, sizeof(float));
		for (c = 0; c < pooll->Channels; c++)
		{
			//将inputData扩大
			matEdgeExpandinf(exInputData, &exInSize, inData[c], &inSize, &psize);
			//Pool(pooll->y[i], &outSize, exInputData, &exInSize, &inSize, pooll->mapSize, pooll->strides, pooll->PoolType);
			for (i = 0; i < outSize.r; i++)
				for (j = 0; j < outSize.c; j++)
				{
					float max = exInputData[i*strides][j*strides];
					for (m = i * strides; m < i*strides + mapSize; m++)
						for (n = j * strides; n < j*strides + mapSize; n++)
							max = exInputData[m][n] > max ? exInputData[m][n] : max;
					pooll->y[c][i][j] = max;
				}
		}
		for (i = 0; i < exInSize.r; i++)
			free(exInputData[i]);
		free(exInputData);
	}
	else if (pooll->PaddingType == valid)
		for (c = 0; c < pooll->Channels; c++)
			//Pool(pooll->y[i], &outSize, inData[i], &inSize, &inSize, pooll->mapSize, pooll->strides, pooll->PoolType);
			for (i = 0; i < outSize.r; i++)
				for (j = 0; j < outSize.c; j++)
				{
					float max = inData[c][i * strides][j * strides];
					for (m = i * strides; m < i*strides + mapSize; m++)
						for (n = j * strides; n < j*strides + mapSize; n++)
							max = inData[c][m][n] > max ? inData[c][m][n] : max;
					pooll->y[c][i][j] = max;
				}
}

void avgpoollff(PoolLayer* pooll, float*** inData)
{
	int i, c, j, m, n;
	int strides = pooll->strides;
	int mapSize = pooll->mapSize;
	nSize inSize = { pooll->inputWidth,pooll->inputHeight };
	nSize outSize = { pooll->outputWidth,pooll->outputHeight };
	padSize psize;
	nSize exInSize;
	if (pooll->PaddingType == same)
	{
		// get padding size
		psize.r = (outSize.r - 1)*pooll->strides + pooll->mapSize - inSize.r;
		psize.c = (outSize.c - 1)*pooll->strides + pooll->mapSize - inSize.c;
		psize.r_top = psize.r / 2;
		psize.r_down = psize.r - psize.r_top;
		psize.c_top = psize.c / 2;
		psize.c_down = psize.c - psize.c_top;
		// extern inputdata & conv
		exInSize.r = psize.r + inSize.r;
		exInSize.c = psize.c + inSize.c;
		float** exInputData = (float**)malloc(exInSize.r * sizeof(float*));
		for (i = 0; i < exInSize.r; i++)
			exInputData[i] = (float*)calloc(exInSize.c, sizeof(float));
		for (c = 0; c < pooll->Channels; c++)
		{
			matEdgeExpand(exInputData, &exInSize, inData[c], &inSize, &psize);
			for (i = 0; i < outSize.r; i++)
				for (j = 0; j < outSize.c; j++)
				{
					float sum = 0.0;
					for (m = i * strides; m < i * strides + mapSize; m++)
						for (n = j * strides; n < j * strides + mapSize; n++)
							sum = sum + exInputData[m][n];

					if (i<psize.r_top)
						sum = sum * mapSize / (float)(mapSize - psize.r_top);
					else if (i*strides+mapSize > psize.r_top+inSize.r)
						sum = sum * mapSize / (float)(mapSize - psize.r_down);
					if (j < psize.c_top)
						sum = sum * mapSize / (float)(mapSize - psize.c_top);
					else if (j*strides + mapSize > psize.c_top + inSize.c)
						sum = sum * mapSize / (float)(mapSize - psize.c_down);
					pooll->y[c][i][j] = sum / (float)(mapSize*mapSize);
				}
		}
		for (i = 0; i < exInSize.r; i++)
			free(exInputData[i]);
		free(exInputData);
	}
	else if (pooll->PaddingType == valid)
		for (c = 0; c < pooll->Channels; c++)
			for (i = 0; i < outSize.r; i++)
				for (j = 0; j < outSize.c; j++)
				{
					float sum = 0.0;
					for (m = i * strides; m < i * strides + mapSize; m++)
						for (n = j * strides; n < j * strides + mapSize; n++)
							sum = sum +inData[c][m][n];
					pooll->y[c][i][j] = sum / (float)(mapSize*mapSize);
				}
}

// the forward propagation of CNN
int cnnff(CNN* cnn, float*** inData)
{
	int result;
	covlff(cnn->C1, inData);
	poollff(cnn->P2, cnn->C1->y);
	covlff(cnn->C3, cnn->P2->y);
	poollff(cnn->P4, cnn->C3->y);
	poollff(cnn->G5, cnn->P4->y);
	covlff_noActi(cnn->F6, cnn->G5->y);
	result = vecmaxIndex(cnn->F6->y, cnn->F6->outChannels);

	return result;
}

int vecmaxIndex(float*** vec, int veclength)// 返回向量最大数的序号
{
	int i;
	float maxnum=-1.0;
	int maxIndex=0;
	for(i=0;i<veclength;i++)
		if(maxnum<vec[i][0][0])
		{
			maxnum=vec[i][0][0];
			maxIndex=i;
		}
	return maxIndex;
}

// load conv layer
void covlloaddir(CovLayer* covl, float**** mapData, float* biasData)
{
	covl->mapData = mapData;
	covl->biasData = biasData;
}

void covlload(CovLayer* covl, FILE *fp)
{
	char* delims = ",";
	char* delims2 = "\n";
	char buffer[1024];
	char *line;
	char *token, *next_token;
	int inc, outc, ii, jj;
	int covl_sz[4];// = { 1, 8, 5, 5 };

	covl_sz[0] = covl->inChannels;
	covl_sz[1] = covl->outChannels;
	covl_sz[2] = covl->mapSize;
	covl_sz[3] = covl->mapSize;
	if (covl_sz[3]>1)
		for (inc = 0; inc < covl_sz[0]; inc++)
			for (outc = 0; outc < covl_sz[1]; outc++)
				for (ii = 0; ii < covl_sz[2]; ii++)
				{
					line = fgets(buffer, sizeof(buffer), fp);
					token = strtok_s(line, delims, &next_token);
					for (jj = 0; jj < covl_sz[3] - 2; jj++)
					{
						covl->mapData[inc][outc][ii][jj] = (float)atof(token);
						token = strtok_s(NULL, delims, &next_token);
					}
					covl->mapData[inc][outc][ii][jj + 1] = (float)atof(token);
					token = strtok_s(NULL, delims2, &next_token);
					covl->mapData[inc][outc][ii][jj + 2] = (float)atof(token);

					//printf("%d %d %d\n", inc,outc,ii);
				}
	else
		for (inc = 0; inc < covl_sz[0]; inc++)
			for (outc = 0; outc < covl_sz[1]; outc++)
			{
				line = fgets(buffer, sizeof(buffer), fp);
				token = strtok_s(line, delims, &next_token);
				covl->mapData[inc][outc][0][0] = (float)atof(token);
			}

	line = fgets(buffer, sizeof(buffer), fp);
	token = strtok_s(line, delims, &next_token);
	for (outc = 0; outc < covl_sz[1] - 2; outc++)
	{
		covl->biasData[outc] = (float)atof(token);
		token = strtok_s(NULL, delims, &next_token);
	}
	covl->biasData[outc + 1] = (float)atof(token);
	token = strtok_s(NULL, delims2, &next_token);
	covl->biasData[outc + 2] = (float)atof(token);
}
void cnnload(CNN* cnn, char *filename)
{
	FILE *fp = NULL;
	errno_t err;

	err = fopen_s(&fp, filename, "r");
	covlload(cnn->C1, fp);
	covlload(cnn->C3, fp);
	covlload(cnn->F6, fp);

	fclose(fp);
}

void get_cov_outsize(CovLayer* covl)
{
	nSize inSize = {covl->inputHeight,covl->inputWidth};

	if (covl->PaddingType==same) // newW = [W/S]
	{
		covl->outputHeight = (int)ceil((inSize.r*1.0) / covl->strides);
		covl->outputWidth = (int)ceil((inSize.c*1.0) / covl->strides);
	}
	else if (covl->PaddingType == valid)// newW = [W-F+1)/S]
	{
		covl->outputHeight = (int)ceil((inSize.r-covl->mapSize+1.0) / covl->strides);
		covl->outputWidth = (int)ceil((inSize.c-covl->mapSize+1.0) / covl->strides);

	}
	else
		printf("error paddingtype\n");
}

CovLayer* initCovLayer(int inputHeight, int inputWidth, int inChannels, CovP* covp)
{
	CovLayer* covL = (CovLayer*)malloc(sizeof(CovLayer));
	covL->inputHeight = inputHeight;
	covL->inputWidth = inputWidth;
	covL->mapSize = covp->mapSize;
	covL->strides = covp->strides;
	covL->inChannels = inChannels;
	covL->outChannels = covp->outChannels;
	covL->PaddingType = covp->PaddingType;
	get_cov_outsize(covL);
	
	//covL->isFullConnect = true; // 默认为全连接

	// the init of matData
	int i, j, c, r;
	covL->mapData = (float****)malloc(inChannels * sizeof(float***));
	for (i = 0; i < inChannels; i++) {
		covL->mapData[i] = (float***)malloc(covL->outChannels * sizeof(float**));
		for (j = 0; j < covL->outChannels; j++) {
			covL->mapData[i][j] = (float**)malloc(covL->mapSize * sizeof(float*));
			for (r = 0; r < covL->mapSize; r++) {
				covL->mapData[i][j][r] = (float*)malloc(covL->mapSize * sizeof(float));
				for (c = 0; c < covL->mapSize; c++)
				{
					covL->mapData[i][j][r][c] = (float)0.0;
				}
			}
		}
	}
	// the init of biasData
	covL->biasData = (float*)calloc(covL->outChannels, sizeof(float));
	// the init of v, y
	int outW = covL->outputWidth;
	int outH = covL->outputHeight;
	covL->v = (float***)malloc(covL->outChannels * sizeof(float**));
	covL->y = (float***)malloc(covL->outChannels * sizeof(float**));
	for (j = 0; j < covL->outChannels; j++) {
		covL->v[j] = (float**)malloc(outH * sizeof(float*));
		covL->y[j] = (float**)malloc(outH * sizeof(float*));
		for (r = 0; r < outH; r++) {
			covL->v[j][r] = (float*)calloc(outW, sizeof(float));
			covL->y[j][r] = (float*)calloc(outW, sizeof(float));
		}
	}

	return covL;
}

void get_pool_outsize(PoolLayer* pooll)
{
	nSize inSize = { pooll->inputHeight,pooll->inputWidth };
	if (pooll->PaddingType == same) // newW = [W/S]
	{
		pooll->outputHeight = (int)ceil((inSize.r*1.0) / pooll->strides);
		pooll->outputWidth = (int)ceil((inSize.c*1.0) / pooll->strides);
	}
	else if (pooll->PaddingType == valid)// newW = [W-F+1)/S]
	{
		pooll->outputHeight = (int)ceil((inSize.r - pooll->mapSize + 1.0) / pooll->strides);
		pooll->outputWidth = (int)ceil((inSize.c - pooll->mapSize + 1.0) / pooll->strides);
	}
	else
		printf("error paddingtype\n");
}

PoolLayer* initPoolLayer(int inputHeight, int inputWidth, int Channels, PoolP* poolp)
{
	PoolLayer* poolL = (PoolLayer*)malloc(sizeof(PoolLayer));
	poolL->inputHeight = inputHeight;
	poolL->inputWidth = inputWidth;
	poolL->mapSize = poolp->mapSize;
	poolL->Channels = Channels;
	poolL->strides = poolp->strides;
	poolL->PaddingType = poolp->PaddingType;
	poolL->PoolType = poolp->pooltype;
	get_pool_outsize(poolL);
	//printf("%d\n", poolL->outputHeight);

	int outW = poolL->outputWidth;
	int outH = poolL->outputHeight;
	int j, r;
	// init y
	poolL->y = (float***)malloc(Channels * sizeof(float**));
	for (j = 0; j < Channels; j++)
	{
		poolL->y[j] = (float**)malloc(outH * sizeof(float*));
		for (r = 0; r < outH; r++)
		{
			poolL->y[j][r] = (float*)calloc(outW, sizeof(float));
		}
	}

	return poolL;
}

FCLayer* initFCLayer(int inputNum, int outputNum)
{
	FCLayer* outL = (FCLayer*)malloc(sizeof(FCLayer));

	outL->inputNum = inputNum;
	outL->outputNum = outputNum;

	outL->biasData = (float*)calloc(outputNum, sizeof(float));

	//outL->d=(float*)calloc(outputNum,sizeof(float));
	outL->v = (float*)calloc(outputNum, sizeof(float));
	outL->y = (float*)calloc(outputNum, sizeof(float));

	// 权重的初始化
	outL->wData = (float**)malloc(outputNum * sizeof(float*)); // 输入行，输出列
	int i, j;
	//srand((unsigned)time(NULL));
	for (i = 0; i < outputNum; i++) {
		outL->wData[i] = (float*)malloc(inputNum * sizeof(float));
		for (j = 0; j < inputNum; j++) {
			//float randnum=(((float)rand()/(float)RAND_MAX)-0.5)*2; // 产生一个-1到1的随机数
			//outL->wData[i][j]=randnum*sqrt((float)6.0/(float)(inputNum+outputNum));
			outL->wData[i][j] = (float)0.0;
		}
	}

	return outL;
}

// 激活函数 input是数据，inputNum说明数据数目，bias表明偏置
float activation_Sigma(float input,float bias) // sigma激活函数
{
	float temp=input+bias;
	return (float)1.0/((float)(1.0+exp(-temp)));
}

// 激活函数 input是数据，inputNum说明数据数目，bias表明偏置
float activation_reLU(float input, float bias)
{
	return input>-bias? input + bias :0;
}

void covlfree(CovLayer* covl)
{
	int i, inc, outc;
	for (inc = 0; inc < covl->inChannels; inc++)
	{ 		
		for (outc = 0; outc < covl->outChannels; outc++)
		{
			for (i = 0; i < covl->mapSize; i++)
			{
				printf("%d\n", i);
				free(covl->mapData[inc][outc][i]);

			}
				
			free(covl->mapData[inc][outc]);
		}
		free(covl->mapData[inc]);
	}
	free(covl->mapData);

	for (outc = 0; outc < covl->outChannels; outc++)
	{
		for (i = 0; i < covl->outputHeight; i++)
		{
			free(covl->v[outc][i]);
			free(covl->y[outc][i]);
		}
		free(covl->v[outc]);
		free(covl->y[outc]);
	}
	free(covl->v);
	free(covl->y);

	free(covl->biasData);
}

