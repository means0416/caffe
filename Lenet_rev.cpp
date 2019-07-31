#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <time.h>
#include <cstring>

#include "Padding.h"
#include "Maxpooling.h"
#include "Convolution.h"
#include "innerProduct.h"
#include "Filter_matrix.h"
#include "memoryTo1stArray.h"
#include "Make_matrix.h"
#include "Activation_ReLu.h"
#include "Memoryfree.h"

using namespace cv;
using namespace std;

int main(){

    double ***i_data, ***conv1, ***conv2, ***pool1, ***pool2, ****filter1, ****filter2, *array3To1, *ip1, *ip2;
    int stride;
    
    printf(" Layer 1 : data, type : Input, params : 1, 1, 28, 28 \n");
    i_data = Make_matrix();
    printf("\n");
    
    printf(" Layer 2 : ConV_1, type : convolution, params : 20, 1, 5, 5 \n"); // pad : 0, stride : 1
    filter1 = Filter_matrix(1); //(필터 선택); 
    conv1 = Convolution(1, 1, 28, 28, 5, 0, i_data, filter1, 20, 1);  //(필터채널, stride, 입력 col크기,  입력 row크기, 필터크기, 패딩크기, 입력 이미지, 필터, 필터개수);
    printf("\n");
    
    printf(" Layer 3 : Pool_1, type : pooling, params : Max, 2, 2 \n"); // stride : 2
    pool1 = Maxpooling(20, 2, 24, 24, 2,  conv1 );  //(입력 채널수, stride, 입력 col크기, 입력 row크기,  필터사이즈,  입력 이미지);
    printf("\n");
    
    printf(" Layer 4 : ConV_2, type : convolution, params : 50, 1, 5, 5 \n"); // pad : 0, stride : 1
    filter2 = Filter_matrix(2); //(필터크기, 필터채널, 필터개수); 
    conv2 = Convolution(20, 1, 12, 12, 5, 0, pool1, filter2, 50, 2); //(필터채널, stride, 입력 col크기,  입력 row크기, 필터크기, 패딩크기, 입력 이미지, 필터, 필터개수);
    printf("\n");

    printf(" Layer 5 : Pool_2, type : pooling, params : Max, 2, 2 \n"); // stride : 2
    pool2 = Maxpooling(50, 2, 8, 8, 2,  conv2); //(입력 채널수, stride, 입력 col크기, 입력 row크기,  필터사이즈,  입력 이미지);
    printf("\n");

    printf(" Layer 6 : Ip_1, type : Inner product, params : 500, 800 \n");
    array3To1 = memoryTo1stArray(pool2,  50, 4, 4); //입력 3차원 메모리,  입력 채널 수, 입력 row크기, 입력 col크기
    ip1 = innerProduct(array3To1, 1); //입력 1차원 메모리, weight 선택
    Activation_ReLu(ip1, 500); //입력 1차원 메모리, 입력 메모리 크기
    printf("\n");

    printf(" Layer 7 : Ip_2, type : Inner product, params : 10, 500 \n");
    ip2 = innerProduct(ip1, 2); //입력 1차원 메모리,  weight 선택
    printf("\n");

    for(int output = 0; output < 10 ; output ++){
        printf("output '%d' value : %.8lf \n",  output , ip2[output]);
    }

    Memoryfree4D( filter1, 20, 1, 5);
    Memoryfree4D( filter2, 50, 20, 5);
    Memoryfree3D( i_data,  1, 5);
    Memoryfree3D( conv1,  20, 24);
    Memoryfree3D( conv2,  50, 8);
    Memoryfree3D( pool1,  20, 12);
    Memoryfree3D( pool2,  50, 4);
    free(ip1);    
    free(ip2);
    free(array3To1);
         
    return 0;
}