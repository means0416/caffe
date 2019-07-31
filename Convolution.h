#include <cstdio>
#include <cstdlib>
#include <iostream>
#include<cmath>
#include<time.h>
using namespace std;

double ***Convolution(int cha_s, int stride, int col_s,  int row_s, int f_size, int p_size, double ***p_data, double ****filter, int filter_qty, int choosebias)
{   
     FILE *file;
    int con_row_size= (row_s - f_size+2*p_size)/stride +1;
    int con_col_size= (col_s - f_size+2*p_size)/stride +1;
    int quantities[1];
    char a[30];
    double ***result;
    double *bias;
    
    if(choosebias == 1){
        file = fopen("Conv1BiasOrigin.txt","r");
    }
    else if(choosebias == 2){
        file = fopen("Conv2BiasOrigin.txt","r");
    }

    if(file == NULL)
    {
        printf("Write Error!!\n");
        return 0;
    }

    fscanf(file, "%d, ", &quantities[0]);
    fscanf(file,"%s", a); 

    bias=(double*)malloc(sizeof(double)*quantities[0]);
    for(int qty = 0 ; qty < quantities[0] ; qty++) {
        fscanf(file," %lf,", &bias[qty]);
        //printf("[%d]. %.3lf \n", qty+1, bias[qty]);  
    }
    result=(double***)malloc(sizeof(double**)*filter_qty);
    for(int cha = 0 ; cha < filter_qty ; cha++){
        *(result+cha)=(double**)malloc(sizeof(double*)*con_row_size);
        for(int row = 0 ; row < con_row_size ; row++){
            *(*(result+cha)+row) = (double*)malloc(sizeof(double)*con_col_size);
                for(int col = 0 ; col<con_col_size ; col++){
                result[cha][row][col] = 0;
            }
        }
    }

    clock_t begin = clock(); // convolution time check!!
    for(int qty_result = 0; qty_result <filter_qty; qty_result++){
        for(int cha_result = 0; cha_result <cha_s; cha_result++){
            for(int row_result = 0; row_result<con_row_size; row_result++){
                for(int col_result = 0; col_result<con_col_size; col_result++){                
                    for(int row_filter = 0; row_filter<f_size; row_filter++){
                        for(int col_filter= 0; col_filter<f_size; col_filter++){
                            result[qty_result][row_result][col_result] += 
                                p_data[cha_result][row_filter+row_result*stride][col_filter+col_result*stride]*filter[qty_result][cha_result][row_filter][col_filter] + bias[qty_result];
                        }    
                    }   
                }
            }
        }
    } 

    clock_t end = clock();
    double time = double(end - begin);
    printf("Convolution sec : %.2lf sec \n", time/CLOCKS_PER_SEC); //CLOCKS_PER_SEC : to check unit second
    free(bias);    
    return result;   
}