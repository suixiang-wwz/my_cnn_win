#ifndef __CNN_
#define __CNN_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "mat.h"

#define AvgPool 0
#define MaxPool 1

// 卷积层
typedef struct convolutional_layer{
	int inputWidth;   //输入图像的宽
	int inputHeight;  //输入图像的长
	int mapSize;      //特征模板的大小，模板一般都是正方形
	int strides;	  //步长

	int inChannels;   //输入图像的数目
	int outChannels;  //输出图像的数目
	int PaddingType;  //same or valid

	int outputWidth;
	int outputHeight;

	// 关于特征模板的权重分布，这里是一个四维数组
	// 其大小为inChannels*outChannels*mapSize*mapSize大小
	float**** mapData;     //存放特征模块的数据
	float* biasData;   //偏置，偏置的大小，为outChannels

	float*** v; // 进入激活函数的输入值
	float*** y; // 激活函数后神经元的输出 
}CovLayer;

// 采样层 pooling
typedef struct pooling_layer{
	int inputWidth;   //输入图像的宽
	int inputHeight;  //输入图像的长
	int mapSize;      //特征模板的大小
	int strides;	  //步长
	int Channels;     //输入图像的数目
	int PaddingType;  //same or valid
	int PoolType;
	int outputWidth;
	int outputHeight;

	float*** y; // 采样函数后神经元的输出,无激活函数
}PoolLayer;

// 输出层 全连接的神经网络
typedef struct fc_layer{
	int inputNum;   //输入数据的数目
	int outputNum;  //输出数据的数目

	float** wData; // 权重数据，为一个inputNum*outputNum大小
	float* biasData;   //偏置，大小为outputNum大小

	float* v; // 进入激活函数的输入值
	float* y; // 激活函数后神经元的输出
}FCLayer;

typedef struct cnn_network{
	CovLayer* C1;
	PoolLayer* P2;
	CovLayer* C3;
	PoolLayer* P4;
	PoolLayer* G5; //global pool layer
	CovLayer* F6; //instead full-connect layer by covl
}CNN;

//the parameter of conv layer
typedef struct convolutional_p { 
	int mapSize;      //特征模板的大小，模板一般都是正方形
	int strides;	  //步长
	int outChannels;
	int PaddingType;  //same or valid
}CovP;
//the parameter of pool layer
typedef struct pooling_p {
	int mapSize;      //特征模板的大小，模板一般都是正方形
	int strides;	  //步长
	int PaddingType;  //same or valid
	int pooltype;
}PoolP;

//functions for init network//////////////////////////////////////////////////////////////
void cnnsetup(CNN* cnn, nSize inputSize, int classes);
// load conv layer
void covlload(CovLayer* covl, FILE *fp);
void covlloaddir(CovLayer* covl, float**** mapData, float* biasData);
void cnnload(CNN* cnn, char *filename);

void get_cov_outsize(CovLayer* covl);
CovLayer* initCovLayer(int inputHeight, int inputWidth, int inChannels, CovP* covp);
void get_pool_outsize(PoolLayer* pooll);
PoolLayer* initPoolLayer(int inputHeight, int inputWidth, int Channels, PoolP* poolp);
FCLayer* initFCLayer(int inputNum,int outputNum);

//functions for forward propagation ////////////////////////////////////////////////////////
// the forward propagation of conv layer
void covlff(CovLayer* covl, float*** inData); 
void covlff_noActi(CovLayer* covl, float*** inData);
// the forward propagation of pool layer
void poollff(PoolLayer* pooll, float*** inData);
void maxpoollff(PoolLayer* pooll, float*** inData);
void avgpoollff(PoolLayer* pooll, float*** inData);
// the forward propagation of cnn
int cnnff(CNN* cnn, float*** inData);

//some other funtions ///////////////////////////////////////////////////////////////////
// 激活函数 input是数据，inputNum说明数据数目，bas表明偏置
float activation_Sigma(float input, float bias); // sigma激活函数
float activation_reLU(float input, float bias);
int vecmaxIndex(float*** vec, int veclength);// 返回向量最大数的序号
void covlfree(CovLayer* covl);// free conv layer

#endif

// global pool
//PoolP G5p = { -1, 1, valid, AvgPool };
//G5p.mapSize = cnn->P4->outputHeight;
//cnn->G5 = initPoolLayer(cnn->P4->outputHeight, cnn->P4->outputWidth, cnn->P4->Channels, &G5p);

