#include <stdlib.h>
#include <string.h>
#include <stdio.h>
//#include <math.h>
//#include <random>
//#include <time.h>
#include "cnn.h"

void cnnsetup(CNN* cnn,nSize inputSize,int outputSize)// insize are all wrong
{
	cnn->layerNum=5;

	nSize inSize;
	int mapSize=5;
	inSize.c=inputSize.c;
	inSize.r=inputSize.r;
	//CovLayer* initCovLayer(int inputWidth, int inputHeight, int mapSize, int strides, int inChannels, int outChannels, int PaddingType)
	cnn->C1= initCovLayer(inSize.c,inSize.r,5,1,1,6,same);
	inSize.c=inSize.c-mapSize+1;
	inSize.r=inSize.r-mapSize+1;
	cnn->S2=initPoolLayer(inSize.c,inSize.r,2,6,6,same,MaxPool);
	inSize.c=inSize.c/2;
	inSize.r=inSize.r/2;
	cnn->C3=initCovLayer(inSize.c,inSize.r,5,1,6,12,same);
	inSize.c=inSize.c-mapSize+1;
	inSize.r=inSize.r-mapSize+1;
	cnn->S4=initPoolLayer(inSize.c,inSize.r,2,12,12,same,MaxPool);
	inSize.c=inSize.c/2;
	inSize.r=inSize.r/2;
	cnn->O5=initFCLayer(inSize.c*inSize.r*12,outputSize);

	//cnn->e=(float*)calloc(cnn->O5->outputNum,sizeof(float));
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
					float max = exInputData[c*strides][j*strides];
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
/*
void cnnff(CNN* cnn, float** inputData)
{
	//int outSizeW=cnn->S2->inputWidth;
	//int outSizeH=cnn->S2->inputHeight;
	int i, j, r, c;
	nSize mapSize = { cnn->C1->mapSize,cnn->C1->mapSize };
	nSize inSize = { cnn->C1->inputWidth,cnn->C1->inputHeight };
	nSize outSize = { cnn->S2->inputWidth,cnn->S2->inputHeight };
	// 第一层的传播
	for (i = 0; i < (cnn->C1->outChannels); i++)
	{
		for (j = 0; j < (cnn->C1->inChannels); j++)
		{
			float** mapout = cov(cnn->C1->mapData[j][i], &mapSize, inputData, &inSize, cnn->C1->strides, same);
			addmat(cnn->C1->v[i], cnn->C1->v[i], outSize, mapout, outSize);
			for (r = 0; r < outSize.r; r++)
				free(mapout[r]);
			free(mapout);
		}
		for (r = 0; r < outSize.r; r++)
			for (c = 0; c < outSize.c; c++)
				cnn->C1->y[i][r][c] = activation_reLU(cnn->C1->v[i][r][c], cnn->C1->biasData[i]);
	}

	// 第二层的输出传播S2，采样层
	outSize.c = cnn->C3->inputWidth;
	outSize.r = cnn->C3->inputHeight;
	inSize.c = cnn->S2->inputWidth;
	inSize.r = cnn->S2->inputHeight;
	for (i = 0; i < (cnn->S2->Channels); i++)
	{
		if (cnn->S2->PaddingType == AvgPool)
			avgPooling(cnn->S2->y[i], outSize, cnn->C1->y[i], inSize, cnn->S2->mapSize);
		else if (cnn->S2->PaddingType == MaxPool)
			maxPooling(cnn->S2->y[i], outSize, cnn->C1->y[i], inSize, cnn->S2->mapSize);
		else
			printf("error when pooling");
	}

	// 第三层输出传播,这里是全连接
	outSize.c = cnn->S4->inputWidth;
	outSize.r = cnn->S4->inputHeight;
	inSize.c = cnn->C3->inputWidth;
	inSize.r = cnn->C3->inputHeight;
	mapSize.c = cnn->C3->mapSize;
	mapSize.r = cnn->C3->mapSize;
	for (i = 0; i < (cnn->C3->outChannels); i++)
	{
		for (j = 0; j < (cnn->C3->inChannels); j++)
		{
			float** mapout = cov(cnn->C3->mapData[j][i], &mapSize, cnn->S2->y[j], &inSize, cnn->C1->strides, same);
			addmat(cnn->C3->v[i], cnn->C3->v[i], outSize, mapout, outSize);
			for (r = 0; r < outSize.r; r++)
				free(mapout[r]);
			free(mapout);
		}
		for (r = 0; r < outSize.r; r++)
			for (c = 0; c < outSize.c; c++)
				cnn->C3->y[i][r][c] = activation_reLU(cnn->C3->v[i][r][c], cnn->C3->biasData[i]);
	}

	// 第四层的输出传播
	inSize.c = cnn->S4->inputWidth;
	inSize.r = cnn->S4->inputHeight;
	outSize.c = inSize.c / cnn->S4->mapSize;
	outSize.r = inSize.r / cnn->S4->mapSize;
	for (i = 0; i < (cnn->S4->Channels); i++) 
	{
		if (cnn->S4->PaddingType == AvgPool)
			avgPooling(cnn->S4->y[i], outSize, cnn->C3->y[i], inSize, cnn->S4->mapSize);
		else if (cnn->S4->PaddingType == MaxPool)
			maxPooling(cnn->S4->y[i], outSize, cnn->C3->y[i], inSize, cnn->S4->mapSize);
		else
			printf("error when pooling");
	}

	// 输出层O5的处理
	// 首先需要将前面的多维输出展开成一维向量
	float* O5inData = (float*)malloc((cnn->O5->inputNum) * sizeof(float));
	for (i = 0; i < (cnn->S4->Channels); i++)
		for (r = 0; r < outSize.r; r++)
			for (c = 0; c < outSize.c; c++)
				O5inData[i*outSize.r*outSize.c + r * outSize.c + c] = cnn->S4->y[i][r][c];

	nSize nnSize = { cnn->O5->inputNum,cnn->O5->outputNum };
	nnff(cnn->O5->v, O5inData, cnn->O5->wData, cnn->O5->biasData, nnSize);
	for (i = 0; i < cnn->O5->outputNum; i++)
		cnn->O5->y[i] = activation_Sigma(cnn->O5->v[i], cnn->O5->biasData[i]);
	free(O5inData);
}
*/

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

// 保存cnn
void savecnn(CNN* cnn, const char* filename)
{
	FILE  *fp=NULL;
	fp=fopen(filename,"wb");
	if(fp==NULL)
		printf("write file failed\n");

	int i,j,r;
	// C1的数据
	for(i=0;i<cnn->C1->inChannels;i++)
		for(j=0;j<cnn->C1->outChannels;j++)
			for(r=0;r<cnn->C1->mapSize;r++)
				fwrite(cnn->C1->mapData[i][j][r],sizeof(float),cnn->C1->mapSize,fp);

	fwrite(cnn->C1->biasData,sizeof(float),cnn->C1->outChannels,fp);

	// C3网络
	for(i=0;i<cnn->C3->inChannels;i++)
		for(j=0;j<cnn->C3->outChannels;j++)
			for(r=0;r<cnn->C3->mapSize;r++)
				fwrite(cnn->C3->mapData[i][j][r],sizeof(float),cnn->C3->mapSize,fp);

	fwrite(cnn->C3->biasData,sizeof(float),cnn->C3->outChannels,fp);

	// O5输出层
	for(i=0;i<cnn->O5->outputNum;i++)
		fwrite(cnn->O5->wData[i],sizeof(float),cnn->O5->inputNum,fp);
	fwrite(cnn->O5->biasData,sizeof(float),cnn->O5->outputNum,fp);

	fclose(fp);
}

// load conv layer
void covlload(CovLayer* covl, float**** mapData, float* biasData)
{
	covl->mapData = mapData;
	covl->biasData = biasData;
}

// 导入cnn的数据
void importcnn(CNN* cnn, const char* filename)
{
	FILE  *fp=NULL;
	fp=fopen(filename,"rb");
	if(fp==NULL)
		printf("write file failed\n");

	int i,j,c,r;
	// C1的数据
	for(i=0;i<cnn->C1->inChannels;i++)
		for(j=0;j<cnn->C1->outChannels;j++)
			for(r=0;r<cnn->C1->mapSize;r++)
				for(c=0;c<cnn->C1->mapSize;c++){
					float* in=(float*)malloc(sizeof(float));
					fread(in,sizeof(float),1,fp);
					cnn->C1->mapData[i][j][r][c]=*in;
				}

	for(i=0;i<cnn->C1->outChannels;i++)
		fread(&cnn->C1->biasData[i],sizeof(float),1,fp);

	// C3网络
	for(i=0;i<cnn->C3->inChannels;i++)
		for(j=0;j<cnn->C3->outChannels;j++)
			for(r=0;r<cnn->C3->mapSize;r++)
				for(c=0;c<cnn->C3->mapSize;c++)
				fread(&cnn->C3->mapData[i][j][r][c],sizeof(float),1,fp);

	for(i=0;i<cnn->C3->outChannels;i++)
		fread(&cnn->C3->biasData[i],sizeof(float),1,fp);

	// O5输出层
	for(i=0;i<cnn->O5->outputNum;i++)
		for(j=0;j<cnn->O5->inputNum;j++)
			fread(&cnn->O5->wData[i][j],sizeof(float),1,fp);

	for(i=0;i<cnn->O5->outputNum;i++)
		fread(&cnn->O5->biasData[i],sizeof(float),1,fp);

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

CovLayer* initCovLayer(int inputHeight, int inputWidth, int mapSize, int strides, int inChannels, int outChannels, int PaddingType)
{
	CovLayer* covL = (CovLayer*)malloc(sizeof(CovLayer));
	covL->inputHeight = inputHeight;
	covL->inputWidth = inputWidth;
	covL->mapSize = mapSize;
	covL->strides = strides;
	covL->inChannels = inChannels;
	covL->outChannels = outChannels;
	covL->PaddingType = PaddingType;
	get_cov_outsize(covL);
	
	//covL->isFullConnect = true; // 默认为全连接

	// the init of matData
	int i, j, c, r;
	covL->mapData = (float****)malloc(inChannels * sizeof(float***));
	for (i = 0; i < inChannels; i++) {
		covL->mapData[i] = (float***)malloc(outChannels * sizeof(float**));
		for (j = 0; j < outChannels; j++) {
			covL->mapData[i][j] = (float**)malloc(mapSize * sizeof(float*));
			for (r = 0; r < mapSize; r++) {
				covL->mapData[i][j][r] = (float*)malloc(mapSize * sizeof(float));
				for (c = 0; c < mapSize; c++) 
				{
					covL->mapData[i][j][r][c] = (float)0.0;
				}
			}
		}
	}
	// the init of biasData
	covL->biasData = (float*)calloc(outChannels, sizeof(float));
	// the init of v, y
	int outW = covL->outputWidth;
	int outH = covL->outputHeight;
	covL->v = (float***)malloc(outChannels * sizeof(float**));
	covL->y = (float***)malloc(outChannels * sizeof(float**));
	for (j = 0; j < outChannels; j++) {
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

PoolLayer* initPoolLayer( int inputHeight, int inputWidth, int mapSize,  int strides, int Channels, int paddingtype,int pooltype)
{
	PoolLayer* poolL = (PoolLayer*)malloc(sizeof(PoolLayer));
	poolL->inputHeight = inputHeight;
	poolL->inputWidth = inputWidth;
	poolL->mapSize = mapSize;
	poolL->Channels = Channels;
	poolL->strides = strides;
	poolL->PaddingType = paddingtype;
	poolL->PoolType = pooltype;
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
	outL->isFullConnect = true;

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



void cnnclear(CNN* cnn)
{
	// 将神经元的部分数据清除
	int j,c,r;
	// C1网络
	for(j=0;j<cnn->C1->outChannels;j++){
		for(r=0;r<cnn->S2->inputHeight;r++){
			for(c=0;c<cnn->S2->inputWidth;c++){
				//cnn->C1->d[j][r][c]=(float)0.0;
				cnn->C1->v[j][r][c]=(float)0.0;
				cnn->C1->y[j][r][c]=(float)0.0;
			}
		}
	}
	// S2网络
	for(j=0;j<cnn->S2->Channels;j++){
		for(r=0;r<cnn->C3->inputHeight;r++){
			for(c=0;c<cnn->C3->inputWidth;c++){
				//cnn->S2->d[j][r][c]=(float)0.0;
				cnn->S2->y[j][r][c]=(float)0.0;
			}
		}
	}
	// C3网络
	for(j=0;j<cnn->C3->outChannels;j++){
		for(r=0;r<cnn->S4->inputHeight;r++){
			for(c=0;c<cnn->S4->inputWidth;c++){
				//cnn->C3->d[j][r][c]=(float)0.0;
				cnn->C3->v[j][r][c]=(float)0.0;
				cnn->C3->y[j][r][c]=(float)0.0;
			}
		}
	}
	// S4网络
	for(j=0;j<cnn->S4->Channels;j++){
		for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
			for(c=0;c<cnn->S4->inputWidth/cnn->S4->mapSize;c++){
				//cnn->S4->d[j][r][c]=(float)0.0;
				cnn->S4->y[j][r][c]=(float)0.0;
			}
		}
	}
	// O5输出
	for(j=0;j<cnn->O5->outputNum;j++){
		//cnn->O5->d[j]=(float)0.0;
		cnn->O5->v[j]=(float)0.0;
		cnn->O5->y[j]=(float)0.0;
	}
}

// 保存CNN网络中的相关数据
void savecnndata(CNN* cnn,const char* filename,float** inputdata)
{
	FILE  *fp=NULL;
	fp=fopen(filename,"wb");
	if(fp==NULL)
		printf("write file failed\n");

	// C1的数据
	int i,j,r;
	// C1网络
	for(i=0;i<cnn->C1->inputHeight;i++)
		fwrite(inputdata[i],sizeof(float),cnn->C1->inputWidth,fp);
	for(i=0;i<cnn->C1->inChannels;i++)
		for(j=0;j<cnn->C1->outChannels;j++)
			for(r=0;r<cnn->C1->mapSize;r++)
				fwrite(cnn->C1->mapData[i][j][r],sizeof(float),cnn->C1->mapSize,fp);

	fwrite(cnn->C1->biasData,sizeof(float),cnn->C1->outChannels,fp);

	for(j=0;j<cnn->C1->outChannels;j++){
		for(r=0;r<cnn->S2->inputHeight;r++){
			fwrite(cnn->C1->v[j][r],sizeof(float),cnn->S2->inputWidth,fp);
		}
		for(r=0;r<cnn->S2->inputHeight;r++){
			//fwrite(cnn->C1->d[j][r],sizeof(float),cnn->S2->inputWidth,fp);
		}
		for(r=0;r<cnn->S2->inputHeight;r++){
			fwrite(cnn->C1->y[j][r],sizeof(float),cnn->S2->inputWidth,fp);
		}
	}

	// S2网络
	for(j=0;j<cnn->S2->Channels;j++){
		for(r=0;r<cnn->C3->inputHeight;r++){
			//fwrite(cnn->S2->d[j][r],sizeof(float),cnn->C3->inputWidth,fp);
		}
		for(r=0;r<cnn->C3->inputHeight;r++){
			fwrite(cnn->S2->y[j][r],sizeof(float),cnn->C3->inputWidth,fp);
		}
	}
	// C3网络
	for(i=0;i<cnn->C3->inChannels;i++)
		for(j=0;j<cnn->C3->outChannels;j++)
			for(r=0;r<cnn->C3->mapSize;r++)
				fwrite(cnn->C3->mapData[i][j][r],sizeof(float),cnn->C3->mapSize,fp);

	fwrite(cnn->C3->biasData,sizeof(float),cnn->C3->outChannels,fp);

	for(j=0;j<cnn->C3->outChannels;j++){
		for(r=0;r<cnn->S4->inputHeight;r++){
			fwrite(cnn->C3->v[j][r],sizeof(float),cnn->S4->inputWidth,fp);
		}
		for(r=0;r<cnn->S4->inputHeight;r++){
			//fwrite(cnn->C3->d[j][r],sizeof(float),cnn->S4->inputWidth,fp);
		}
		for(r=0;r<cnn->S4->inputHeight;r++){
			fwrite(cnn->C3->y[j][r],sizeof(float),cnn->S4->inputWidth,fp);
		}
	}

	// S4网络
	for(j=0;j<cnn->S4->Channels;j++){
		for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
			//fwrite(cnn->S4->d[j][r],sizeof(float),cnn->S4->inputWidth/cnn->S4->mapSize,fp);
		}
		for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
			fwrite(cnn->S4->y[j][r],sizeof(float),cnn->S4->inputWidth/cnn->S4->mapSize,fp);
		}
	}

	// O5输出层
	for(i=0;i<cnn->O5->outputNum;i++)
		fwrite(cnn->O5->wData[i],sizeof(float),cnn->O5->inputNum,fp);
	fwrite(cnn->O5->biasData,sizeof(float),cnn->O5->outputNum,fp);
	fwrite(cnn->O5->v,sizeof(float),cnn->O5->outputNum,fp);
	//fwrite(cnn->O5->d,sizeof(float),cnn->O5->outputNum,fp);
	fwrite(cnn->O5->y,sizeof(float),cnn->O5->outputNum,fp);

	fclose(fp);
}

/*
// 测试cnn函数
float cnntest(CNN* cnn, ImgArr inputData,LabelArr outputData,int testNum)
{
	int n=0;
	int incorrectnum=0;  //错误预测的数目
	for(n=0;n<testNum;n++){
		cnnff(cnn,inputData->ImgPtr[n].ImgData);
		if(vecmaxIndex(cnn->O5->y,cnn->O5->outputNum)!=vecmaxIndex(outputData->LabelPtr[n].LabelData,cnn->O5->outputNum))
			incorrectnum++;
		cnnclear(cnn);
	}
	return (float)incorrectnum/(float)testNum;
}
*/

/*
void cnntrain(CNN* cnn,	ImgArr inputData,LabelArr outputData,CNNOpts opts,int trainNum)
{
	// 学习训练误差曲线
	cnn->L=(float*)malloc(trainNum*sizeof(float));
	int e;
	for(e=0;e<opts.numepochs;e++){
		int n=0;
		for(n=0;n<trainNum;n++){
			//printf("%d\n",n);
			cnnff(cnn,inputData->ImgPtr[n].ImgData);  // 前向传播，这里主要计算各
			cnnbp(cnn,outputData->LabelPtr[n].LabelData); // 后向传播，这里主要计算各神经元的误差梯度


			char* filedir="E:\\Code\\Matlab\\PicTrans\\CNNData\\";
			const char* filename=combine_strings(filedir,combine_strings(intTochar(n),".cnn"));
			savecnndata(cnn,filename,inputData->ImgPtr[n].ImgData);
			cnnapplygrads(cnn,opts,inputData->ImgPtr[n].ImgData); // 更新权重

			cnnclear(cnn);
			// 计算并保存误差能量
			float l=0.0;
			int i;
			for(i=0;i<cnn->O5->outputNum;i++)
				l=l+cnn->e[i]*cnn->e[i];
			if(n==0)
				cnn->L[n]=l/(float)2.0;
			else
				cnn->L[n]=cnn->L[n-1]*0.99+0.01*l/(float)2.0;
		}
	}
}
*/

/*
float sigma_derivation(float y){ // Logic激活函数的自变量微分
	return y*(1-y); // 这里y是指经过激活函数的输出值，而不是自变量
}
*/
/*
void cnnbp(CNN* cnn,float* outputData) // 网络的后向传播
{
	int i,j,c,r; // 将误差保存到网络中
	for(i=0;i<cnn->O5->outputNum;i++)
		cnn->e[i]=cnn->O5->y[i]-outputData[i];

	//从后向前反向计算
	// 输出层O5
	for(i=0;i<cnn->O5->outputNum;i++)
		cnn->O5->d[i]=cnn->e[i]*sigma_derivation(cnn->O5->y[i]);

	// S4层，传递到S4层的误差
	// 这里没有激活函数
	nSize outSize={cnn->S4->inputWidth/cnn->S4->mapSize,cnn->S4->inputHeight/cnn->S4->mapSize};
	for(i=0;i<cnn->S4->outChannels;i++)
		for(r=0;r<outSize.r;r++)
			for(c=0;c<outSize.c;c++)
				for(j=0;j<cnn->O5->outputNum;j++){
					int wInt=i*outSize.c*outSize.r+r*outSize.c+c;
					cnn->S4->d[i][r][c]=cnn->S4->d[i][r][c]+cnn->O5->d[j]*cnn->O5->wData[j][wInt];
				}

	// C3层
	// 由S4层传递的各反向误差,这里只是在S4的梯度上扩充一倍
	int mapdata=cnn->S4->mapSize;
	nSize S4dSize={cnn->S4->inputWidth/cnn->S4->mapSize,cnn->S4->inputHeight/cnn->S4->mapSize};
	// 这里的Pooling是求平均，所以反向传递到下一神经元的误差梯度没有变化
	for(i=0;i<cnn->C3->outChannels;i++){
		float** C3e=UpSample(cnn->S4->d[i],S4dSize,cnn->S4->mapSize,cnn->S4->mapSize);
		for(r=0;r<cnn->S4->inputHeight;r++)
			for(c=0;c<cnn->S4->inputWidth;c++)
				cnn->C3->d[i][r][c]=C3e[r][c]*sigma_derivation(cnn->C3->y[i][r][c])/(float)(cnn->S4->mapSize*cnn->S4->mapSize);
		for(r=0;r<cnn->S4->inputHeight;r++)
			free(C3e[r]);
		free(C3e);
	}

	// S2层，S2层没有激活函数，这里只有卷积层有激活函数部分
	// 由卷积层传递给采样层的误差梯度，这里卷积层共有6*12个卷积模板
	outSize.c=cnn->C3->inputWidth;
	outSize.r=cnn->C3->inputHeight;
	nSize inSize={cnn->S4->inputWidth,cnn->S4->inputHeight};
	nSize mapSize={cnn->C3->mapSize,cnn->C3->mapSize};
	for(i=0;i<cnn->S2->outChannels;i++){
		for(j=0;j<cnn->C3->outChannels;j++){
			float** corr=correlation(cnn->C3->mapData[i][j],mapSize,cnn->C3->d[j],inSize,full);
			addmat(cnn->S2->d[i],cnn->S2->d[i],outSize,corr,outSize);
			for(r=0;r<outSize.r;r++)
				free(corr[r]);
			free(corr);
		}

		//for(r=0;r<cnn->C3->inputHeight;r++)
			//for(c=0;c<cnn->C3->inputWidth;c++)
				// 这里本来用于采样的激活

	}

	// C1层，卷积层
	mapdata=cnn->S2->mapSize;
	nSize S2dSize={cnn->S2->inputWidth/cnn->S2->mapSize,cnn->S2->inputHeight/cnn->S2->mapSize};
	// 这里的Pooling是求平均，所以反向传递到下一神经元的误差梯度没有变化
	for(i=0;i<cnn->C1->outChannels;i++){
		float** C1e=UpSample(cnn->S2->d[i],S2dSize,cnn->S2->mapSize,cnn->S2->mapSize);
		for(r=0;r<cnn->S2->inputHeight;r++)
			for(c=0;c<cnn->S2->inputWidth;c++)
				cnn->C1->d[i][r][c]=C1e[r][c]*sigma_derivation(cnn->C1->y[i][r][c])/(float)(cnn->S2->mapSize*cnn->S2->mapSize);
		for(r=0;r<cnn->S2->inputHeight;r++)
			free(C1e[r]);
		free(C1e);
	}
}
*/
/*
void cnnapplygrads(CNN* cnn,CNNOpts opts,float** inputData) // 更新权重
{
	// 这里存在权重的主要是卷积层和输出层
	// 更新这两个地方的权重就可以了
	int i,j,r,c;

	// C1层的权重更新
	nSize dSize={cnn->S2->inputHeight,cnn->S2->inputWidth};
	nSize ySize={cnn->C1->inputHeight,cnn->C1->inputWidth};
	nSize mapSize={cnn->C1->mapSize,cnn->C1->mapSize};

	for(i=0;i<cnn->C1->outChannels;i++){
		for(j=0;j<cnn->C1->inChannels;j++){
			float** flipinputData=rotate180(inputData,ySize);
			float** C1dk=cov(cnn->C1->d[i],dSize,flipinputData,ySize,valid);
			multifactor(C1dk,C1dk,mapSize,-1*opts.alpha);
			addmat(cnn->C1->mapData[j][i],cnn->C1->mapData[j][i],mapSize,C1dk,mapSize);
			for(r=0;r<(dSize.r-(ySize.r-1));r++)
				free(C1dk[r]);
			free(C1dk);
			for(r=0;r<ySize.r;r++)
				free(flipinputData[r]);
			free(flipinputData);
		}
		cnn->C1->biasData[i]=cnn->C1->biasData[i]-opts.alpha*summat(cnn->C1->d[i],dSize);
	}

	// C3层的权重更新
	dSize.c=cnn->S4->inputWidth;
	dSize.r=cnn->S4->inputHeight;
	ySize.c=cnn->C3->inputWidth;
	ySize.r=cnn->C3->inputHeight;
	mapSize.c=cnn->C3->mapSize;
	mapSize.r=cnn->C3->mapSize;
	for(i=0;i<cnn->C3->outChannels;i++){
		for(j=0;j<cnn->C3->inChannels;j++){
			float** flipinputData=rotate180(cnn->S2->y[j],ySize);
			float** C3dk=cov(cnn->C3->d[i],dSize,flipinputData,ySize,valid);
			multifactor(C3dk,C3dk,mapSize,-1.0*opts.alpha);
			addmat(cnn->C3->mapData[j][i],cnn->C3->mapData[j][i],mapSize,C3dk,mapSize);
			for(r=0;r<(dSize.r-(ySize.r-1));r++)
				free(C3dk[r]);
			free(C3dk);
			for(r=0;r<ySize.r;r++)
				free(flipinputData[r]);
			free(flipinputData);
		}
		cnn->C3->biasData[i]=cnn->C3->biasData[i]-opts.alpha*summat(cnn->C3->d[i],dSize);
	}

	// 输出层
	// 首先需要将前面的多维输出展开成一维向量
	float* O5inData=(float*)malloc((cnn->O5->inputNum)*sizeof(float));
	nSize outSize={cnn->S4->inputWidth/cnn->S4->mapSize,cnn->S4->inputHeight/cnn->S4->mapSize};
	for(i=0;i<(cnn->S4->outChannels);i++)
		for(r=0;r<outSize.r;r++)
			for(c=0;c<outSize.c;c++)
				O5inData[i*outSize.r*outSize.c+r*outSize.c+c]=cnn->S4->y[i][r][c];

	for(j=0;j<cnn->O5->outputNum;j++){
		for(i=0;i<cnn->O5->inputNum;i++)
			cnn->O5->wData[j][i]=cnn->O5->wData[j][i]-opts.alpha*cnn->O5->d[j]*O5inData[i];
		cnn->O5->basicData[j]=cnn->O5->basicData[j]-opts.alpha*cnn->O5->d[j];
	}
	free(O5inData);
}
*/

// 单层全连接神经网络的前向传播
float vecMulti(float* vec1, float* vec2, int vecL)// 两向量相乘
{
	int i;
	float m = 0;
	for (i = 0; i < vecL; i++)
		m = m + vec1[i] * vec2[i];
	return m;
}

void nnff(float* output, float* input, float** wdata, float* bas, nSize nnSize)
{
	int w = nnSize.c;
	int h = nnSize.r;

	int i;
	for (i = 0; i < h; i++)
		output[i] = vecMulti(input, wdata[i], w) + bas[i];
}