//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!

#include "fft.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// you may define other parameters here!
// you may define other macros here!
// you may define other functions here!

__device__ unsigned int RightIndex(unsigned int num,unsigned int M ) //making right indexes
{ 
	unsigned int right_num = 0; 
	int temp;
	
    for ( int i = 0; i < M; i++)
    {
        temp = (num & (1 << i));
        if(temp){
			right_num = right_num |(1 << ((M - 1) - i));
		}
    }
  
    return right_num;
}

//-----------------------------------------------------------------------------
__global__ void kernelFunc(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M,unsigned int i) 
{   

    float tempr,tempi;
	
	unsigned int ind = bx*blockDim.x+tx;
    unsigned int k_butt = ind%(1<<(M-1-i));
    unsigned int k_part = (unsigned int)(ind/(1<<(M-1-i)));
    unsigned int butter_ind = 2*(1<<i)*k_butt + k_part;
    unsigned int w = k_part*(1<<(M-1-i));
  
           
    if (i==0) {
		
		tempr = cos((2*PI*w)/N)*x_r_d[RightIndex(butter_ind + 1<<i,M)] + sin((2*PI*w)/N)*x_i_d[RightIndex(butter_ind + 1<<i,M)];
		tempi = cos((2*PI*w)/N)*x_i_d[RightIndex(butter_ind + 1<<i,M)] - sin((2*PI*w)/N)*x_r_d[RightIndex(butter_ind + 1<<i,M)];
	 
		X_r_d[butter_ind] = x_r_d[RightIndex(butter_ind,M)] + tempr;
		X_i_d[butter_ind] = x_i_d[RightIndex(butter_ind,M)] + tempi;
		
		X_r_d[butter_ind + 1<<i] = x_r_d[RightIndex(butter_ind,M)] - tempr;
		X_i_d[butter_ind + 1<<i] = x_i_d[RightIndex(butter_ind,M)] - tempi;
    }
    else {
		
		tempr = cos((2*PI*w)/N)*x_r_d[butter_ind + 1<<i] + sin((2*PI*w)/N)*x_i_d[butter_ind + 1<<i]; //real part of multiplication
		tempi = cos((2*PI*w)/N)*x_i_d[butter_ind + 1<<i] - sin((2*PI*w)/N)*x_r_d[butter_ind + 1<<i]; //imaginary part of multiplication
	 
		X_r_d[butter_ind] = x_r_d[butter_ind] + tempr;
		X_i_d[butter_ind] = x_i_d[butter_ind] + tempi;
		
		X_r_d[butter_ind + 1<<i] = x_r_d[butter_ind] - tempr;
		X_i_d[butter_ind + 1<<i] = x_i_d[butter_ind] - tempi; 
    }
    

}
//-----------------------------------------------------------------------------
__global__ void kernelFunc2(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M) 
{
	//...
}
//-----------------------------------------------------------------------------
void gpuKernel_simple(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M)
{
	// In this function, both inputs and outputs are on GPU.
	// No need for cudaMalloc, cudaMemcpy or cudaFree.

    int a,b;
    
    float* image,*real;
    real = (float*) malloc(N * sizeof(float));
    image = (float*) malloc(N * sizeof(float));
    
    
    if (M>=11) { a=1024; b=N/2048;}
    else {a=N/2;b=1;}

	dim3 dimGrid(b,1,1);
	dim3 dimBlock(a,1,1);
	int k;

    for(k=0;k<M;++k){
        if (k % 2 == 0 ){
        kernelFunc <<< dimGrid, dimBlock >>> (x_r_d, x_i_d, X_r_d, X_i_d, N, M,k);
        
        HANDLE_ERROR(cudaMemcpy(real, X_r_d, N * sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(image, X_i_d, N * sizeof(float), cudaMemcpyDeviceToHost));
        
        }
        else{
        kernelFunc <<< dimGrid, dimBlock >>> (X_r_d, X_i_d, x_r_d, x_i_d, N, M,k);
        
        HANDLE_ERROR(cudaMemcpy(real, x_r_d, N * sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(image, x_i_d, N * sizeof(float), cudaMemcpyDeviceToHost));
        
        }
    }
    if (M%2==0){
    HANDLE_ERROR(cudaMemcpy(X_r_d, x_r_d, N * sizeof(float), cudaMemcpyDeviceToDevice));
    HANDLE_ERROR(cudaMemcpy(X_i_d, x_i_d, N * sizeof(float), cudaMemcpyDeviceToDevice));    
    }

    HANDLE_ERROR(cudaMemcpy(real, X_r_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(image, X_i_d, N * sizeof(float), cudaMemcpyDeviceToHost));
	free(real);
	free(image);

}
//-----------------------------------------------------------------------------
void gpuKernel_efficient(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M)
{
	// In this function, both inputs and outputs are on GPU.
	// No need for cudaMalloc, cudaMemcpy or cudaFree.
	
	dim3 dimGrid(1,1);
	dim3 dimBlock(1,1);

	kernelFunc2 <<< dimGrid, dimBlock >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M);
}
