#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <iostream>
#include<cmath>
#include<cstring>
using namespace cv;
using namespace std;

double ****Filter_matrix(int filterchoose) 
{
    FILE *file;
    int quantities[1], channel[1], rowsize[1], colsize[1];
    char a[30]; // 더미 배열
    if( filterchoose == 1){
        file = fopen("Conv1WeightOrigin.txt","r");
    }
    else if( filterchoose == 2){
        file = fopen("Conv2WeightOrigin.txt","r");
    }

    if(file == NULL)
    {
        printf("Write Error!!\n");
        return 0;
    }

    fscanf(file, "%d, %d, %d, %d, ", &quantities[0], &channel[0], &rowsize[0], &colsize[0]);
    printf(" Q`ty : %d, Channel : %d, Row : %d, Column : %d", quantities[0], channel[0], rowsize[0], colsize[0]);
    fscanf(file,"%s", a);    

    double  ****weight;
    weight = (double****)malloc(sizeof(double***)*quantities[0]);
    for(int qty = 0; qty < quantities[0]; qty ++){
        *(weight+qty) = (double***)malloc(sizeof(double**)*channel[0]);
        for(int cha = 0 ; cha < channel[0]; cha++){
            *(*(weight+qty)+cha) = (double**)malloc(sizeof(double*)*rowsize[0]);
            for(int row = 0 ; row < rowsize[0]; row++){
                *(*(*(weight+qty)+cha)+row) = (double*)malloc(sizeof(double)*colsize[0]);
                for(int col = 0 ; col < colsize[0]; col++){
                     fscanf(file," %lf,", &weight[qty][cha][row][col]);
                 }
            }
        }
    }

    return weight;
}