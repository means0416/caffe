#include <cstdio>
#include <cstdlib>
#include <iostream>
using namespace std;

double ***Padding255(int channelSize, int padSize, int columnSize, int rowSize, int ***input ) // d : input data Qty , n : input data size
{

    double ***result;
    int columnResult = columnSize+(2*padSize); //data size
    int rowResult = rowSize+(2*padSize); //data size

    result = (double***)malloc(channelSize*sizeof(double**));
    for(int cha = 0; cha < channelSize; cha++)
    {   
        *(result+cha) = (double**)malloc(rowSize*sizeof(double*));
        for(int row = 0; row < rowSize ; row++) 
         {
           *(*(result+cha)+row) = (double*)malloc(columnSize*sizeof(double));
           for(int col = 0; col < columnSize ; col++)
           {
                if((padSize<=row)&&(row<rowSize+padSize)&&(padSize<=col)&&(col<columnSize+padSize))
                {
                    result[cha][row][col] = input[cha][row-padSize][col-padSize];  
                }else  {
                    result[cha][row][col] = 255;
                }
           }
        }
    }
    return result;
}

double ***Padding0(int channelSize, int padSize, int columnSize, int rowSize, int ***input ) // d : input data Qty , n : input data size
{

    double ***result;
    int columnResult = columnSize+(2*padSize); //data size
    int rowResult = rowSize+(2*padSize); //data size

    result = (double***)malloc(channelSize*sizeof(double**));
    for(int cha = 0; cha < channelSize; cha++)
    {   
        *(result+cha) = (double**)malloc(rowSize*sizeof(double*));
        for(int row = 0; row < rowSize ; row++) 
         {
           *(*(result+cha)+row) = (double*)malloc(columnSize*sizeof(double));
           for(int col = 0; col < columnSize ; col++)
           {
                if((padSize<=row)&&(row<rowSize+padSize)&&(padSize<=col)&&(col<columnSize+padSize))
                {
                    result[cha][row][col] = input[cha][row-padSize][col-padSize];  
                }else  {
                    result[cha][row][col] = 0;
                }
           }
        }
    }
    return result;
}