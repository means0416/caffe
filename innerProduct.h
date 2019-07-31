#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <iostream>
#include<time.h>
#include<cstring>
using namespace cv;
using namespace std;

double *innerProduct(double *input,  int weightchoose){
    FILE *weightFile, *biasFile;
    char a[30]; // 더미 배열
    int quantities[1], weights[1], biasquantities[1];
    if(weightchoose == 1)
    {
        weightFile=fopen("Ip1WeightOrigin.txt","r");
        biasFile=fopen("Ip1BiasOrigin.txt","r");
    }
    else if(weightchoose == 2)
    {
        weightFile=fopen("Ip2WeightOrigin.txt","r");
        biasFile=fopen("Ip2BiasOrigin.txt","r");
    }

    if(weightFile == NULL)
    {
        printf("weight file write Error!!\n");
        return 0;
    }

    if(biasFile == NULL)
    {
        printf("bias file write Error!!\n");
        return 0;
    }    

    fscanf(weightFile,"%d, %d, ",&quantities[0], &weights[0]);
    printf(" output : %d EA, input : %d EA \n", quantities[0], weights[0]);
    fscanf(weightFile, "%s", a);    

    double **backpropResult = (double**)malloc(sizeof(double*)*quantities[0]);
    for(int qty = 0; qty < quantities[0]; qty++){
        *(backpropResult+qty) = (double*)malloc(sizeof(double)*weights[0]);
        for(int w = 0; w < weights[0]; w++){
            fscanf(weightFile, " %lf,", &backpropResult[qty][w]);
        }
    }

    fscanf(biasFile,"%d, ", &biasquantities[0]);
    printf(" bias Q`ty = %d EA \n", biasquantities[0]);
    fscanf(biasFile, "%s", a);
    double *ipBias = (double*)malloc(sizeof(double)*biasquantities[0]);
    for(int qty = 0; qty < biasquantities[0]; qty++){
        fscanf(biasFile," %lf,", &ipBias[qty]);
    } 

    double *result = (double*)malloc(sizeof(double)*quantities[0]);
    for(int out = 0 ; out<quantities[0] ; out++){
            result[out] = 0; 
    }    

    for(int out = 0 ; out<quantities[0] ; out++){
        for(int in = 0 ; in<weights[0] ; in++ ){
            result[out] += input[in]*backpropResult[out][in] + ipBias[out]; 
        }
    }

    for(int qty = 0; qty < quantities[0]; qty++){
        free(*(backpropResult+qty));
    }free(backpropResult);
    return result;
}