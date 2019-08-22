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
	// ��������ά���飬��Ҫ��Ϊ�˱���ȫ���ӵ���ʽ��ʵ���Ͼ���㲢û���õ�ȫ���ӵ���ʽ
	float**** mapData;     //�������ģ�������
	float* biasData;   //ƫ�ã�ƫ�õĴ�С��ΪoutChannels

	// �������ߵĴ�Сͬ�����ά����ͬ
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

	// �������ߵĴ�Сͬ�����ά����ͬ
	float* v; // ���뼤���������ֵ
	float* y; // ���������Ԫ�����
	//float* d; // ����ľֲ��ݶ�,��ֵ

	bool isFullConnect; //�Ƿ�Ϊȫ����
}FCLayer;

typedef struct cnn_network{
	int layerNum;
	CovLayer* C1;
	PoolLayer* S2;
	CovLayer* C3;
	PoolLayer* S4;
	FCLayer* O5;

	float* e; // ѵ�����
	float* L; // ˲ʱ�������
}CNN;

typedef struct train_opts{
	int numepochs; // ѵ���ĵ�������
	float alpha; // ѧϰ����
}CNNOpts;

void cnnsetup(CNN* cnn,nSize inputSize,int outputSize);

// ����cnn
void savecnn(CNN* cnn, const char* filename);
// load conv layer
void covlload(CovLayer* covl, float**** mapData, float* biasData);
// ����cnn������
void importcnn(CNN* cnn, const char* filename);

// ��ʼ�������
void get_cov_outsize(CovLayer* covl);
CovLayer* initCovLayer(int inputHeight, int inputWidth, int mapSize, int strides, int inChannels, int outChannels, int PaddingType);
// ��ʼ��������
void get_pool_outsize(PoolLayer* pooll);
PoolLayer* initPoolLayer(int inputHeight, int inputWidth, int mapSize, int strides, int Channels, int paddingtype, int pooltype);
// ��ʼ�������
FCLayer* initFCLayer(int inputNum,int outputNum);

// ����� input�����ݣ�inputNum˵��������Ŀ��bas����ƫ��
float activation_Sigma(float input,float bias); // sigma�����
float activation_reLU(float input, float bias);

// the forward propagation of conv layer
void covlff(CovLayer* covl, float*** inData); 
void poollff(PoolLayer* pooll, float*** inData);
// the forward propagation of pool layer
void maxpoollff(PoolLayer* pooll, float*** inData);
void avgpoollff(PoolLayer* pooll, float*** inData);


void cnnff(CNN* cnn,float** inputData); // �����ǰ�򴫲�
//void cnnbp(CNN* cnn,float* outputData); // ����ĺ��򴫲�
//void cnnapplygrads(CNN* cnn,CNNOpts opts,float** inputData);
void cnnclear(CNN* cnn); // ������vyd����
int vecmaxIndex(float*** vec, int veclength);// ������������������
/* 
	����ȫ����������Ĵ���
	nnSize������Ĵ�С
*/
void nnff(float* output,float* input,float** wdata,float* bas,nSize nnSize); // ����ȫ�����������ǰ�򴫲�

void savecnndata(CNN* cnn,const char* filename,float** inputdata); // ����CNN�����е��������

#endif
