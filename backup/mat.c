#include <stdlib.h>
//#include <string.h>
#include <stdio.h>
#include <math.h>
//#include <random>
//#include <time.h>
#include "mat.h"

float** create_mat(nSize matSize)
{
	int ii = 0, jj = 0;
	float **mat = (float**)malloc(sizeof(float*) * matSize.c);
	for (ii = 0; ii < matSize.c; ii++)
	{
		mat[ii] = (float*)malloc(sizeof(float) * matSize.r);
	}
	return mat;
}

int printmat(float** mat, nSize matSize)
{
	int ii = 0, jj = 0;
	for (ii = 0; ii < matSize.c; ii++)
	{
		for (jj = 0; jj < matSize.r; jj++)
		{
			printf("%.2f,", *(*(mat + ii) + jj));
		}
		printf("\n");
	}
	return 0;
}

float** rotate180(float** mat, nSize matSize)// 矩阵翻转180度
{
	int i,c,r;
	int outSizeW=matSize.c;
	int outSizeH=matSize.r;
	float** outputData=(float**)malloc(outSizeH*sizeof(float*));
	for(i=0;i<outSizeH;i++)
		outputData[i]=(float*)malloc(outSizeW*sizeof(float));

	for(r=0;r<outSizeH;r++)
		for(c=0;c<outSizeW;c++)
			outputData[r][c]=mat[outSizeH-r-1][outSizeW-c-1];

	return outputData;
}

// 关于卷积和相关操作的输出选项
// 这里共有三种选择：full、same、valid，分别表示
// full指完全，操作后结果的大小为inSize+(mapSize-1)
// same指同输入相同大小
// valid指完全操作后的大小，一般为inSize-(mapSize-1)大小，其不需要将输入添0扩大。

float** cov(float** map,nSize* mapSize,float** inputData,nSize* inSize, int strides,int type)// 互相关
{
	// map must be square
	int i,j,c,r;
	padSize psize;
	nSize outSize;
	nSize exInSize;
	float** outputData;
	if (type==same)// newW = [W/S]
	{
		outSize.r = (int)ceil((inSize->c*1.0) / strides);
		outSize.c = (int)ceil((inSize->r*1.0) / strides);
		// get padding size
		psize.r = (outSize.r - 1)*strides + mapSize->r - inSize->r;
		psize.c = (outSize.c - 1)*strides + mapSize->c - inSize->c;
		psize.r_top = psize.r / 2;
		psize.r_down = psize.r - psize.r_top;
		psize.c_top = psize.c / 2;
		psize.c_down = psize.c - psize.c_top;

		exInSize.r = psize.r + inSize->r;
		exInSize.c = psize.c + inSize->c;
		// define outputData
		outputData = (float**)malloc(outSize.r * sizeof(float*));
		for (i = 0; i < outSize.r; i++)
			outputData[i] = (float*)calloc(outSize.c, sizeof(float));

		float** exInputData = (float**)malloc(exInSize.r * sizeof(float*));
		for (i = 0; i < exInSize.r; i++)
			exInputData[i] = (float*)calloc(exInSize.c, sizeof(float));

		// inputData扩大
		matEdgeExpand(exInputData, &exInSize, inputData, inSize, &psize);
		// conv
		for (j = 0; j < outSize.r; j++)
			for (i = 0; i < outSize.c; i = i++)
				for (r = 0; r < mapSize->r; r++)
					for (c = 0; c < mapSize->c; c++)
					{
						outputData[j][i] = outputData[j][i] + map[r][c] * exInputData[j*strides + r][i*strides + c];
					}
		for (i = 0; i < exInSize.c; i++)
			free(exInputData[i]);
		free(exInputData);
	}
	else if (type==valid)//newW = [W - F + 1) / S]
	{
		outSize.r = (int)ceil((inSize->r - mapSize->r + 1.0) / strides);
		outSize.c = (int)ceil((inSize->c - mapSize->c + 1.0) / strides);
		// define outputData
		outputData = (float**)malloc(outSize.r * sizeof(float*));
		for (i = 0; i < outSize.r; i++)
			outputData[i] = (float*)calloc(outSize.c, sizeof(float));
		// conv
		for (j = 0; j < outSize.r; j++)
			for (i = 0; i < outSize.c; i = i++)
				for (r = 0; r < mapSize->r; r++)
					for (c = 0; c < mapSize->c; c++)
					{
						outputData[j][i] = outputData[j][i] + map[r][c] * inputData[j*strides + r][i*strides + c];
					}
	}
	else
	{
		outSize.r = (int)ceil((inSize->c*1.0) / strides);
		outSize.c = (int)ceil((inSize->r*1.0) / strides);
		// define outputData
		outputData = (float**)malloc(outSize.r * sizeof(float*));
		for (i = 0; i < outSize.r; i++)
			outputData[i] = (float*)calloc(outSize.c, sizeof(float));
	}
	return outputData;
}
/*
float** cov(float** map,nSize mapSize,float** inputData,nSize inSize,int type) // 卷积操作
{
	// 卷积操作可以用旋转180度的特征模板相关来求
	float** flipmap=rotate180(map,mapSize); //旋转180度的特征模板
	//float** res=correlation(flipmap,mapSize,inputData,inSize,type);
	float** res = correlation(map, mapSize, inputData, inSize, type);
	int i;
	for(i=0;i<mapSize.r;i++)
		free(flipmap[i]);
	free(flipmap);
	return res;
}
*/

// 这个是矩阵的上采样（等值内插），upc及upr是内插倍数
float** UpSample(float** mat,nSize matSize,int upc,int upr)
{ 
	int i,j,m,n;
	int c=matSize.c;
	int r=matSize.r;
	float** res=(float**)malloc((r*upr)*sizeof(float*)); // 结果的初始化
	for(i=0;i<(r*upr);i++)
		res[i]=(float*)malloc((c*upc)*sizeof(float));

	for(j=0;j<r*upr;j=j+upr){
		for(i=0;i<c*upc;i=i+upc)// 宽的扩充
			for(m=0;m<upc;m++)
				res[j][i+m]=mat[j/upr][i/upc];

		for(n=1;n<upr;n++)      //  高的扩充
			for(i=0;i<c*upc;i++)
				res[j+n][i]=res[j][i];
	}
	return res;
}

// 给二维矩阵边缘扩大，增加addw大小的0值边
void matEdgeExpand(float** outmat, nSize* outSize, float** mat,nSize* matSize,padSize* psize)
{ // 向量边缘扩大
	int i,j;
	int c=matSize->c;
	int r=matSize->r;

	for (j = 0; j < outSize->r; j++)
	{
		for (i = 0; i < outSize->c; i++)
		{
			if (j < psize->r_top || i < psize->c_top || j >= (r + psize->r_top) || i >= (c + psize->c_top))
				outmat[j][i] = (float)0.0;
			else
				outmat[j][i] = mat[j - psize->r_top][i - psize->c_top]; // 复制原向量的数据
		}
	}

}

void matEdgeExpandinf(float** outmat, nSize* outSize, float** mat, nSize* matSize, padSize* psize)
{ // 向量边缘扩大
	int i, j;
	int c = matSize->c;
	int r = matSize->r;

	for (j = 0; j < outSize->r; j++)
	{
		for (i = 0; i < outSize->c; i++)
		{
			if (j < psize->r_top || i < psize->c_top || j >= (r + psize->r_top) || i >= (c + psize->c_top))
				outmat[j][i] = (float)-99999;
			else
				outmat[j][i] = mat[j - psize->r_top][i - psize->c_top]; // 复制原向量的数据
		}
	}

}

/*  backup of matEdgeExpand
float** matEdgeExpand(float** mat,nSize* matSize,padSize* psize)
{ // 向量边缘扩大
	int i,j;
	int c=matSize.c;
	int r=matSize.r;
	float** res;
	if (addc % 2 == 1)
	{
		res = (float**)malloc((r + addc-1) * sizeof(float*)); // 结果的初始化
		for (i = 0; i < (r + addc-1); i++)
			res[i] = (float*)malloc((c + addc -1) * sizeof(float));

		for (j = 0; j < r + addc-1; j++)
		{
			for (i = 0; i < c + addc-1; i++)
			{
				if (j<(addc-1)/2 || i<(addc-1)/2 || j>=(r+(addc-1)/2) || i>=(c+(addc-1)/2))
					res[j][i] = (float)0.0;
				else
					res[j][i] = mat[j - (addc-1)/2][i - (addc-1)/2]; // 复制原向量的数据
			}
		}

	}
	else
	{
		res = (float**)malloc((matSize.r + addc) * sizeof(float*));
		for (i = 0; i < (matSize.r + addc); i++)
			res[i] = (float*)malloc((matSize.c + addc) * sizeof(float));

		for (j = 0; j < matSize.r + addc; j++)
		{
			for (i = 0; i < matSize.c + addc; i++)
			{
				if (j<(addc/2-1) || i<(addc/2-1) || j>=(r+(addc/2-1)) || i>=(c+(addc/2-1)))
					res[j][i] = (float)0.0;
				else
					res[j][i] = mat[j - (addc/2 - 1)][i - (addc/2 - 1)]; // 复制原向量的数据
			}
		}
	}

	return res;
}
*/
// 给二维矩阵边缘缩小，擦除shrinkc大小的边
float** matEdgeShrink(float** mat,nSize matSize,int shrinkc,int shrinkr) // 向量的缩小，宽缩小addw，高缩小addh
{ 
	int i,j;
	int c=matSize.c;
	int r=matSize.r;
	float** res=(float**)malloc((r-2*shrinkr)*sizeof(float*)); // 结果矩阵的初始化
	for(i=0;i<(r-2*shrinkr);i++)
		res[i]=(float*)malloc((c-2*shrinkc)*sizeof(float));

	
	for(j=0;j<r;j++){
		for(i=0;i<c;i++){
			if(j>=shrinkr&&i>=shrinkc&&j<(r-shrinkr)&&i<(c-shrinkc))
				res[j-shrinkr][i-shrinkc]=mat[j][i]; // 复制原向量的数据
		}
	}
	return res;
}

void savemat(float** mat,nSize matSize,const char* filename)
{
	FILE  *fp=NULL;
	fp=fopen(filename,"w");
	if(fp==NULL)
		printf("write file failed\n");

	int i;
	for(i=0;i<matSize.r;i++)
		fwrite(mat[i],sizeof(float),matSize.c,fp);
	fclose(fp);
}

void addmat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2)// 矩阵相加
{
	int i,j;
	if(matSize1.c!=matSize2.c||matSize1.r!=matSize2.r)
		printf("ERROR: Size is not same!");

	for(i=0;i<matSize1.r;i++)
		for(j=0;j<matSize1.c;j++)
			res[i][j]=mat1[i][j]+mat2[i][j];
}

void multifactor(float** res, float** mat, nSize matSize, float factor)// 矩阵乘以系数
{
	int i,j;
	for(i=0;i<matSize.r;i++)
		for(j=0;j<matSize.c;j++)
			res[i][j]=mat[i][j]*factor;
}

float summat(float** mat,nSize matSize) // 矩阵各元素的和
{
	float sum=0.0;
	int i,j;
	for(i=0;i<matSize.r;i++)
		for(j=0;j<matSize.c;j++)
			sum=sum+mat[i][j];
	return sum;
}