#include <cstdio>
#include <cstdlib>
#include <iostream>
#include<cmath>
using namespace std;

double ***Maxpooling(int cha_s, int stride, int col_s, int row_s,  int f_size,  double ***i_data)
{
    int pool_col_size = (col_s-f_size)/stride +1;
    int pool_row_size = (row_s-f_size)/stride +1;
    double ***result;
    int max = INT_MIN;

    result=(double***)malloc(sizeof(double**)*cha_s);
    for(int cha = 0 ; cha < cha_s ; cha++){
        *(result+cha)=(double**)malloc(sizeof(double*)*pool_row_size);
        for(int row = 0 ; row < pool_row_size ; row++){
            *(*(result+cha)+row) = (double*)malloc(sizeof(double)*pool_col_size);
        }
    }
    
    for(int cha = 0 ; cha <cha_s ; cha++){ // Result reset
        for(int row = 0 ; row<pool_row_size ; row++){
            for(int col = 0 ; col<pool_col_size ; col++){
                result[cha][row][col] = 0;
            }
        }
    }

    for(int pool_cha = 0; pool_cha <cha_s; pool_cha++){
        for(int pool_row = 0; pool_row<pool_row_size; pool_row++){
            for(int pool_col = 0; pool_col<pool_col_size; pool_col++){
                for(int filter_row = 0; filter_row<f_size; filter_row++){
                    for(int filter_col= 0; filter_col<f_size; filter_col++){
                        if(i_data[pool_cha][filter_row+pool_row*stride][filter_col+pool_col*stride]>max){
                            max = i_data[pool_cha][filter_row+pool_row*stride][filter_col+pool_col*stride];
                        }
                    }
                } 
                result[pool_cha][pool_row][pool_col] = max;
                max = INT_MIN;
            } 
        }
    }
    return result;
}