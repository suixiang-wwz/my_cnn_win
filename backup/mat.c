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

float** rotate180(float** mat, nSize matSize)// ����ת180��
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

// ���ھ������ز��������ѡ��
// ���ﹲ������ѡ��full��same��valid���ֱ��ʾ
// fullָ��ȫ�����������Ĵ�СΪinSize+(mapSize-1)
// sameָͬ������ͬ��С
// validָ��ȫ������Ĵ�С��һ��ΪinSize-(mapSize-1)��С���䲻��Ҫ��������0����

float** cov(float** map,nSize* mapSize,float** inputData,nSize* inSize, int strides,int type)// �����
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

		// inputData����
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
float** cov(float** map,nSize mapSize,float** inputData,nSize inSize,int type) // �������
{
	// ���������������ת180�ȵ�����ģ���������
	float** flipmap=rotate180(map,mapSize); //��ת180�ȵ�����ģ��
	//float** res=correlation(flipmap,mapSize,inputData,inSize,type);
	float** res = correlation(map, mapSize, inputData, inSize, type);
	int i;
	for(i=0;i<mapSize.r;i++)
		free(flipmap[i]);
	free(flipmap);
	return res;
}
*/

// ����Ǿ�����ϲ�������ֵ�ڲ壩��upc��upr���ڲ屶��
float** UpSample(float** mat,nSize matSize,int upc,int upr)
{ 
	int i,j,m,n;
	int c=matSize.c;
	int r=matSize.r;
	float** res=(float**)malloc((r*upr)*sizeof(float*)); // ����ĳ�ʼ��
	for(i=0;i<(r*upr);i++)
		res[i]=(float*)malloc((c*upc)*sizeof(float));

	for(j=0;j<r*upr;j=j+upr){
		for(i=0;i<c*upc;i=i+upc)// �������
			for(m=0;m<upc;m++)
				res[j][i+m]=mat[j/upr][i/upc];

		for(n=1;n<upr;n++)      //  �ߵ�����
			for(i=0;i<c*upc;i++)
				res[j+n][i]=res[j][i];
	}
	return res;
}

// ����ά�����Ե��������addw��С��0ֵ��
void matEdgeExpand(float** outmat, nSize* outSize, float** mat,nSize* matSize,padSize* psize)
{ // ������Ե����
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
				outmat[j][i] = mat[j - psize->r_top][i - psize->c_top]; // ����ԭ����������
		}
	}

}

void matEdgeExpandinf(float** outmat, nSize* outSize, float** mat, nSize* matSize, padSize* psize)
{ // ������Ե����
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
				outmat[j][i] = mat[j - psize->r_top][i - psize->c_top]; // ����ԭ����������
		}
	}

}

/*  backup of matEdgeExpand
float** matEdgeExpand(float** mat,nSize* matSize,padSize* psize)
{ // ������Ե����
	int i,j;
	int c=matSize.c;
	int r=matSize.r;
	float** res;
	if (addc % 2 == 1)
	{
		res = (float**)malloc((r + addc-1) * sizeof(float*)); // ����ĳ�ʼ��
		for (i = 0; i < (r + addc-1); i++)
			res[i] = (float*)malloc((c + addc -1) * sizeof(float));

		for (j = 0; j < r + addc-1; j++)
		{
			for (i = 0; i < c + addc-1; i++)
			{
				if (j<(addc-1)/2 || i<(addc-1)/2 || j>=(r+(addc-1)/2) || i>=(c+(addc-1)/2))
					res[j][i] = (float)0.0;
				else
					res[j][i] = mat[j - (addc-1)/2][i - (addc-1)/2]; // ����ԭ����������
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
					res[j][i] = mat[j - (addc/2 - 1)][i - (addc/2 - 1)]; // ����ԭ����������
			}
		}
	}

	return res;
}
*/
// ����ά�����Ե��С������shrinkc��С�ı�
float** matEdgeShrink(float** mat,nSize matSize,int shrinkc,int shrinkr) // ��������С������Сaddw������Сaddh
{ 
	int i,j;
	int c=matSize.c;
	int r=matSize.r;
	float** res=(float**)malloc((r-2*shrinkr)*sizeof(float*)); // �������ĳ�ʼ��
	for(i=0;i<(r-2*shrinkr);i++)
		res[i]=(float*)malloc((c-2*shrinkc)*sizeof(float));

	
	for(j=0;j<r;j++){
		for(i=0;i<c;i++){
			if(j>=shrinkr&&i>=shrinkc&&j<(r-shrinkr)&&i<(c-shrinkc))
				res[j-shrinkr][i-shrinkc]=mat[j][i]; // ����ԭ����������
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

void addmat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2)// �������
{
	int i,j;
	if(matSize1.c!=matSize2.c||matSize1.r!=matSize2.r)
		printf("ERROR: Size is not same!");

	for(i=0;i<matSize1.r;i++)
		for(j=0;j<matSize1.c;j++)
			res[i][j]=mat1[i][j]+mat2[i][j];
}

void multifactor(float** res, float** mat, nSize matSize, float factor)// �������ϵ��
{
	int i,j;
	for(i=0;i<matSize.r;i++)
		for(j=0;j<matSize.c;j++)
			res[i][j]=mat[i][j]*factor;
}

float summat(float** mat,nSize matSize) // �����Ԫ�صĺ�
{
	float sum=0.0;
	int i,j;
	for(i=0;i<matSize.r;i++)
		for(j=0;j<matSize.c;j++)
			sum=sum+mat[i][j];
	return sum;
}