#include <cstdio>
#include <iostream>
using namespace std;

void Activation_ReLu(double *input, int in_qty){
    for(int in = 0; in<in_qty; in++)
    {   
        if(input[in]<= 0){
            input[in] = 0;
        }        
    }
}