#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace std;

double *memoryTo1stArray(double ***pooling,  int channel_in, int row_in, int col_in){
    int temp = 0;

    double *result = (double*)malloc(sizeof(double)*(channel_in * row_in * col_in));

    for(int cha = 0 ; cha<channel_in ; cha++ ){
        for(int row = 0 ; row<row_in ; row++){
            for(int col = 0; col<col_in ; col++){
                    result[cha*row_in*col_in + row*col_in + col] = pooling[cha][row][col]; 
            }
        }
    }

    return result;
}