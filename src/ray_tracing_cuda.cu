#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <assert.h>

#define nthreads_per_block 512 // Number of threads per block
//#define nthreads_per_block 256 // Number of threads per block
//#define nthreads_per_block 128 // Number of threads per block
#define PI 3.14159265

struct vector {
    float x, y, z;
};

/* Initializes the curandState for each thread */
__global__ void setup_states(curandState *states, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, id, &states[id]);
}

__global__ void initCurandStates(curandState *states, int numStates) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < numStates) {
        // Use id * 4238811 as the seed for each state for uniqueness
        curand_init(id * 4238811ULL, 0, 0, &states[id]);
    }
}

void write_to_file(float *G, int n, char *file_name){
	FILE *fp = fopen(file_name, "w");
	for (int i=(n-1); i>=0; i--){
		for(int j=0; j<n; j++)
		 	fprintf(fp, "%f ", G[j + (n * i)]);
		fprintf(fp, "\n");
	}
	fclose(fp);
}

__global__ void ray_tracing(curandState *states, float *G, float W_max, float W_y, float R, float L_x, float L_y, float L_z, float C_y, int n) {
    
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = states[id];

    /* Predefined */
    vector W = {0.0f, W_y, 0.0f};
    vector C = {0.0f, C_y, 0.0f};
    vector L = {L_x, L_y, L_z};

    vector V,I,N,S;
    
    for(int ray=0; ray<10;ray++){ 
        float temp = -1.0f;
        while ((fabs(W.x) > W_max) || (fabs(W.z) > W_max) || temp <= 0.0f) {

            float phi = 1.0f * PI * curand_uniform(&localState);
            float cosTheta = 2.0f * curand_uniform(&localState) - 1.0f;
            float sinTheta = sqrt(1.0f - (cosTheta * cosTheta));

            V.x = sinTheta * (float) cos(phi);
            V.y = sinTheta * (float) sin(phi);
            V.z = cosTheta;

            if (V.y != 0.0f) {
                W.x = (W.y / V.y) * V.x;
                W.z = (W.y / V.y) * V.z;

                float vc = (V.x * C.x) + (V.y * C.y) + (V.z * C.z);
                float cc = (C.x * C.x) + (C.y * C.y) + (C.z * C.z);
                temp = (vc * vc) + (R * R) - cc;
            }
        }

        //printf("W.x: %f, W.y:%f, W.z:%f\n", W.x, W.y, W.z);

        float t = (V.x * C.x) + (V.y * C.y) + (V.z * C.z) - sqrt(temp);
        I.x = t * V.x;
        I.y = t * V.y;
        I.z = t * V.z;

        N.x = (I.x - C.x) / sqrt((I.x - C.x)*(I.x - C.x) + (I.y - C.y)*(I.y - C.y) + (I.z - C.z)*(I.z - C.z));
        N.y = (I.y - C.y) / sqrt((I.x - C.x)*(I.x - C.x) + (I.y - C.y)*(I.y - C.y) + (I.z - C.z)*(I.z - C.z));
        N.z = (I.z - C.z) / sqrt((I.x - C.x)*(I.x - C.x) + (I.y - C.y)*(I.y - C.y) + (I.z - C.z)*(I.z - C.z));

        S.x = (L.x - I.x) / sqrt((L.x - I.x)*(L.x - I.x) + (L.y - I.y)*(L.y - I.y) + (L.z - I.z)*(L.z - I.z));
        S.y = (L.y - I.y) / sqrt((L.x - I.x)*(L.x - I.x) + (L.y - I.y)*(L.y - I.y) + (L.z - I.z)*(L.z - I.z));
        S.z = (L.z - I.z) / sqrt((L.x - I.x)*(L.x - I.x) + (L.y - I.y)*(L.y - I.y) + (L.z - I.z)*(L.z - I.z));

        float b = (S.x * N.x) + (S.y * N.y) + (S.z * N.z);
        if (b<0.0f)
            b = 0.0f;

        float normalizedWx = (W.x + W_max) / (2.0f * W_max);
        int i_index = (int)(normalizedWx * n);
        if (i_index < 0) i_index = 0;
        if (i_index >= n) i_index = n - 1;

        float normalizedWz = (W.z + W_max) / (2.0f * W_max);
        int j_index = (int)(normalizedWz * n);
        if (j_index < 0) j_index = 0;
        if (j_index >= n) j_index = n - 1;

        //assert( (i_index < n) && (i_index>=0) );
        //assert( (j_index < n) && (j_index>=0) );

        atomicAdd(&G[j_index + (n * i_index)], b);
        //printf("G: %f\n", G[j_index + (n * i_index)]);
    }
}


int main() {
    int n = 1000; // Total number of threads
    long int N_rays = 1000000000;
    //int nblocks = (N_rays + nthreads_per_block - 1) / nthreads_per_block;
    
    int nblocks = (N_rays / (10 * nthreads_per_block)) + 1;
    //size_t size_per_state = sizeof(curandState);
    //size_t total_size = size_per_state * totalStates;
    //printf("Size per state: %zu\nTotal required size for states: %zu\n", size_per_state, total_size);


    //printf("nblocks:%d\n", nblocks);

    float W_max = 2.0f, W_y = 2.0f, R = 6.0f, C_y = 12.0f, L_x = 4.0f, L_y = 4.0f, L_z = -1.0f;

    /* CUDA timers */
    cudaEvent_t start_device, stop_device;  
    float time_device;

    /* creates CUDA timers but does not start yet */
    cudaEventCreate(&start_device);
    cudaEventCreate(&stop_device);

    /* initialize data on host */
	float *h_G = (float *) malloc(sizeof(float) * n * n);
    for (int i=0; i<(n*n); i++)
		h_G[i] = 0.0f;
    assert(h_G);
		
	float *d_G;
    curandState *d_states;

    /* allocate device memory */
	cudaError_t err = cudaMalloc((void **) &d_G, sizeof(float) * n * n);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector G (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	cudaMemcpy(d_G, h_G, sizeof(float) * n * n, cudaMemcpyHostToDevice);

    cudaError_t cudaStatus;
    /* Initialize cuRAND states */ 
    cudaStatus = cudaMalloc((void **) &d_states, (N_rays/10) * sizeof(curandState));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for cuRAND states: %s\n", cudaGetErrorString(cudaStatus));
        // Add any necessary cleanup here
        return 1; // or exit(EXIT_FAILURE);
    }
    
    //printf("cudaMalloc status d_states: %s\n", cudaGetErrorString(cudaGetLastError()));
    //setup_states<<<nblocks, nthreads_per_block>>>(d_states, time(NULL));
    initCurandStates<<<nblocks, nthreads_per_block>>>(d_states, (N_rays/10));
    cudaDeviceSynchronize(); // Ensure setup_states kernel finishes before proceeding
    //printf("synchronize status States: %s\n", cudaGetErrorString(cudaGetLastError())); 
    
    //printf("launching kernel with %d blocks, %d threads per block\n", nblocks, nthreads_per_block);
    cudaEventRecord( start_device, 0 );  
  	ray_tracing<<<nblocks,nthreads_per_block>>>(d_states, d_G, W_max, W_y, R, L_x, L_y, L_z, C_y, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch ray_tracing kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    //printf("kernel: %s\n", cudaGetErrorString(cudaGetLastError())); 

    cudaEventRecord( stop_device, 0 );
    cudaEventSynchronize( stop_device );
    //printf("synchronize status G: %s\n", cudaGetErrorString(cudaGetLastError())); 
    cudaEventElapsedTime( &time_device, start_device, stop_device );

    /* copy data back to memory */
    cudaMemcpy(h_G,d_G,n*n*sizeof(float), cudaMemcpyDeviceToHost);
    //printf("cudaMemcpy status: %s\n", cudaGetErrorString(cudaGetLastError()));

    //printf("G:%f\n", d_G[1000*500+1000]); 
    //printf("G:%f\n", h_G[1000*500+1000]); 

    printf("time elapsed device: %f(s)\n",  time_device/1000.);
    
    /* Save results to a file */
    char *filename = "cuda_sphere.txt";
    write_to_file(h_G, n, filename);

    free(h_G);
    cudaFree(d_G);
    cudaFree(d_states);
    cudaEventDestroy( start_device );
    cudaEventDestroy( stop_device );

}

