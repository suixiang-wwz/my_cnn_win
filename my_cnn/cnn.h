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

// �����
typedef struct convolutional_layer{
	int inputWidth;   //����ͼ��Ŀ�
	int inputHeight;  //����ͼ��ĳ�
	int mapSize;      //����ģ��Ĵ�С��ģ��һ�㶼��������
	int strides;	  //����

	int inChannels;   //����ͼ�����Ŀ
	int outChannels;  //���ͼ�����Ŀ
	int PaddingType;  //same or valid

	int outputWidth;
	int outputHeight;

	// ��������ģ���Ȩ�طֲ���������һ����ά����
	// ���СΪinChannels*outChannels*mapSize*mapSize��С
	float**** mapData;     //�������ģ�������
	float* biasData;   //ƫ�ã�ƫ�õĴ�С��ΪoutChannels

	float*** v; // ���뼤���������ֵ
	float*** y; // ���������Ԫ����� 
}CovLayer;

// ������ pooling
typedef struct pooling_layer{
	int inputWidth;   //����ͼ��Ŀ�
	int inputHeight;  //����ͼ��ĳ�
	int mapSize;      //����ģ��Ĵ�С
	int strides;	  //����
	int Channels;     //����ͼ�����Ŀ
	int PaddingType;  //same or valid
	int PoolType;
	int outputWidth;
	int outputHeight;

	float*** y; // ������������Ԫ�����,�޼����
}PoolLayer;

// ����� ȫ���ӵ�������
typedef struct fc_layer{
	int inputNum;   //�������ݵ���Ŀ
	int outputNum;  //������ݵ���Ŀ

	float** wData; // Ȩ�����ݣ�Ϊһ��inputNum*outputNum��С
	float* biasData;   //ƫ�ã���СΪoutputNum��С

	float* v; // ���뼤���������ֵ
	float* y; // ���������Ԫ�����
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
	int mapSize;      //����ģ��Ĵ�С��ģ��һ�㶼��������
	int strides;	  //����
	int outChannels;
	int PaddingType;  //same or valid
}CovP;
//the parameter of pool layer
typedef struct pooling_p {
	int mapSize;      //����ģ��Ĵ�С��ģ��һ�㶼��������
	int strides;	  //����
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
// ����� input�����ݣ�inputNum˵��������Ŀ��bas����ƫ��
float activation_Sigma(float input, float bias); // sigma�����
float activation_reLU(float input, float bias);
int vecmaxIndex(float*** vec, int veclength);// ������������������
void covlfree(CovLayer* covl);// free conv layer

#endif

// global pool
//PoolP G5p = { -1, 1, valid, AvgPool };
//G5p.mapSize = cnn->P4->outputHeight;
//cnn->G5 = initPoolLayer(cnn->P4->outputHeight, cnn->P4->outputWidth, cnn->P4->Channels, &G5p);

