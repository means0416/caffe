#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace std;

void Memoryfree4D( double ****memory, int quantities, int channel, int rowsize){
        for(int qty = 0 ; qty < quantities ; qty++){    
        for(int cha = 0; cha < channel ; cha++){
            for(int row = 0; row < rowsize ; row++) {
                free(*(*(*(memory+qty)+cha)+row));
            }free(*(*(memory+qty)+cha));
        }free(*(memory+qty));
    }free(memory);
}

void Memoryfree3D( double ***memory,  int channel, int rowsize){
    for(int cha = 0; cha <channel ; cha++) {
        for(int row = 0; row < rowsize ; row++){
            free(*(*(memory+cha)+row));
        } free(*(memory+cha));
    }free(memory);
}

void Memoryfree2D( double **memory,  int rowsize){
    for(int row = 0; row < rowsize ; row++){
            free(*(memory+row));
        } free(memory);
}