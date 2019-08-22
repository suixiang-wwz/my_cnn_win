#ifndef __CNN_
#define __CNN_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "mat.h"
//#include <random>
//#include <time.h>
//#include "minst.h"


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
	// 这里用四维数组，主要是为了表现全连接的形式，实际上卷积层并没有用到全连接的形式
	float**** mapData;     //存放特征模块的数据
	float* biasData;   //偏置，偏置的大小，为outChannels

	// 下面三者的大小同输出的维度相同
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

	// 下面三者的大小同输出的维度相同
	float* v; // 进入激活函数的输入值
	float* y; // 激活函数后神经元的输出
	//float* d; // 网络的局部梯度,δ值

	bool isFullConnect; //是否为全连接
}FCLayer;

typedef struct cnn_network{
	int layerNum;
	CovLayer* C1;
	PoolLayer* S2;
	CovLayer* C3;
	PoolLayer* S4;
	FCLayer* O5;

	float* e; // 训练误差
	float* L; // 瞬时误差能量
}CNN;

typedef struct train_opts{
	int numepochs; // 训练的迭代次数
	float alpha; // 学习速率
}CNNOpts;

void cnnsetup(CNN* cnn,nSize inputSize,int outputSize);

// 保存cnn
void savecnn(CNN* cnn, const char* filename);
// load conv layer
void covlload(CovLayer* covl, float**** mapData, float* biasData);
// 导入cnn的数据
void importcnn(CNN* cnn, const char* filename);

// 初始化卷积层
void get_cov_outsize(CovLayer* covl);
CovLayer* initCovLayer(int inputHeight, int inputWidth, int mapSize, int strides, int inChannels, int outChannels, int PaddingType);
// 初始化采样层
void get_pool_outsize(PoolLayer* pooll);
PoolLayer* initPoolLayer(int inputHeight, int inputWidth, int mapSize, int strides, int Channels, int paddingtype, int pooltype);
// 初始化输出层
FCLayer* initFCLayer(int inputNum,int outputNum);

// 激活函数 input是数据，inputNum说明数据数目，bas表明偏置
float activation_Sigma(float input,float bias); // sigma激活函数
float activation_reLU(float input, float bias);

// the forward propagation of conv layer
void covlff(CovLayer* covl, float*** inData); 
void poollff(PoolLayer* pooll, float*** inData);
// the forward propagation of pool layer
void maxpoollff(PoolLayer* pooll, float*** inData);
void avgpoollff(PoolLayer* pooll, float*** inData);


void cnnff(CNN* cnn,float** inputData); // 网络的前向传播
//void cnnbp(CNN* cnn,float* outputData); // 网络的后向传播
//void cnnapplygrads(CNN* cnn,CNNOpts opts,float** inputData);
void cnnclear(CNN* cnn); // 将数据vyd清零
int vecmaxIndex(float*** vec, int veclength);// 返回向量最大数的序号
/* 
	单层全连接神经网络的处理
	nnSize是网络的大小
*/
void nnff(float* output,float* input,float** wdata,float* bas,nSize nnSize); // 单层全连接神经网络的前向传播

void savecnndata(CNN* cnn,const char* filename,float** inputdata); // 保存CNN网络中的相关数据

#endif
