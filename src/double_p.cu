#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <assert.h>

__constant__ double W_y = 2;
__constant__ double W_max = 2;
__constant__ double R = 6;
__constant__ double PI = 3.14159265;

/*
struct vector {
    double x, y, z;
};
*/


typedef struct vectData {
	double x;
	double y;
	double z;
} vector;

__device__ vector create_vector(double x, double y, double z)
{
	vector vec;
	vec.x = x;
	vec.y = y;
	vec.z = z;
	return vec;
}



__global__ void initCurandStates(curandState *states, int numStates) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < numStates) {
        // Use id * 4238811 as the seed for each state for uniqueness
        curand_init(id * 4238811ULL, 0, 0, &states[id]);
    }
}

void write_to_file(float *G, int n, char *file_name) {
    FILE *fp = fopen(file_name, "wb"); // Open file in binary write mode
    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    for (int i = (n-1); i >= 0; i--) {
        // Write a row of n floats to the file as binary data
        if (fwrite(&G[i*n], sizeof(float), n, fp) != n) {
            perror("Error writing to file");
            break;
        }
    }
    fclose(fp);
}

__global__ void ray_tracing(float *G, curandState *states, int *samples, int N_rays, int n, int total_threads, double L_x, double L_y, double L_z, double C_y) {
    
    //printf("total_threads:%d\n", total_threads);
    //printf("N_rays:%d\n", total_threads);
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    curandState localstate = states[id];
    
    //int stride = blockDim.x * gridDim.x;   //total threads (nblocks*ntpb)

    /* Predefined */
    vector L = create_vector(L_x, L_y, L_z);	
    vector C = create_vector(0, C_y, 0);	
    vector W = create_vector(0, W_y, 0);	

    //vector W = {0, W_y, 0};
    //vector C = {0, C_y, 0};
    //vector L = {L_x, L_y, L_z};

    //printf("size:%d\n",W.y);

    vector V,I,N,S;

    //printf("N_rays:%d\n", N_rays);
    //printf("total_threads:%d\n", total_threads);
        
    for(int ray = id; ray < N_rays; ray += total_threads){
        

        double temp = -1;
        int s = 0;
        
        while ((fabs(W.x) > W_max) || (fabs(W.z) > W_max) || temp <= 0.0) {
        //while ((W.x > W_max) || (W.x < -1*W_max) || (W.z > W_max) || (W.z < -1*W_max)|| temp <= 0) {
            
            double phi = 1 * PI * curand_uniform(&localstate);
            double cosTheta = 2 * curand_uniform(&localstate) - 1;
            double sinTheta = sqrt(1 - (cosTheta * cosTheta));

            V.x = sinTheta * (double) cos(phi);
            V.y = sinTheta * (double) sin(phi);
            V.z = cosTheta;

            s+=1;

            if (V.y != 0) {
                W.x = (W.y / V.y) * V.x;
                W.z = (W.y / V.y) * V.z;

                double vc = (V.x * C.x) + (V.y * C.y) + (V.z * C.z);
                double cc = (C.x * C.x) + (C.y * C.y) + (C.z * C.z);
                temp = (vc * vc) + (R * R) - cc;
            }
        }

        //printf("W.x %f,W.z%f,temp%f\n",W.x,W.z,temp );

        double t = (V.x * C.x) + (V.y * C.y) + (V.z * C.z) - sqrt(temp);
        I.x = t * V.x;
        I.y = t * V.y;
        I.z = t * V.z;

        N.x = (I.x - C.x) / sqrt((I.x - C.x)*(I.x - C.x) + (I.y - C.y)*(I.y - C.y) + (I.z - C.z)*(I.z - C.z));
        N.y = (I.y - C.y) / sqrt((I.x - C.x)*(I.x - C.x) + (I.y - C.y)*(I.y - C.y) + (I.z - C.z)*(I.z - C.z));
        N.z = (I.z - C.z) / sqrt((I.x - C.x)*(I.x - C.x) + (I.y - C.y)*(I.y - C.y) + (I.z - C.z)*(I.z - C.z));

        S.x = (L.x - I.x) / sqrt((L.x - I.x)*(L.x - I.x) + (L.y - I.y)*(L.y - I.y) + (L.z - I.z)*(L.z - I.z));
        S.y = (L.y - I.y) / sqrt((L.x - I.x)*(L.x - I.x) + (L.y - I.y)*(L.y - I.y) + (L.z - I.z)*(L.z - I.z));
        S.z = (L.z - I.z) / sqrt((L.x - I.x)*(L.x - I.x) + (L.y - I.y)*(L.y - I.y) + (L.z - I.z)*(L.z - I.z));

        double b = (S.x * N.x) + (S.y * N.y) + (S.z * N.z);
        if (b<0)
            b = 0;

        double normalizedWx = (W.x + W_max) / (2 * W_max);
        int i_index = (int)(normalizedWx * n);
        if (i_index < 0) i_index = 0;
        if (i_index >= n) i_index = n - 1;

        double normalizedWz = (W.z + W_max) / (2 * W_max);
        int j_index = (int)(normalizedWz * n);
        if (j_index < 0) j_index = 0;
        if (j_index >= n) j_index = n - 1;

        assert( (i_index < n) && (i_index>=0) );
        assert( (j_index < n) && (j_index>=0) );

        //printf("B:%f\n", (float) b);

        atomicAdd(&G[j_index + (n * i_index)], (float) b);
        //printf("s:%d\n",s);
        *samples+=s;
        //printf("G: %f\n", G[j_index + (n * i_index)]);
    }
    // bug fix, set the state again to ensure continuity 
    states[id] = localstate;
}


int main(int argc, char * argv[]) {

    if (argc != 5)
	{
		printf("4 arguments needed: number of rays, number of grid points, number of blocks, and threads per block. Exiting.\n");
		return 0;
	}

    clock_t start_total, end_total;
    start_total = clock();

    int N_rays = atoi(argv[1]);
    int n = atoi(argv[2]);
    int nblocks = atoi(argv[3]);
    int ntpb = atoi(argv[4]);

    int samples = 0;
    int *d_samples;

    cudaMalloc( (void **) &d_samples, sizeof(int));
    cudaMemcpy(d_samples, &samples, sizeof(int), cudaMemcpyHostToDevice);


    int total_threads = nblocks * ntpb;
    
    double C_y = 12, L_x = 4, L_y = 4, L_z = -1;

    float *d_G, *h_G;
    size_t size = n * n * sizeof(float);

    /* initialize data on host */
    h_G = (float *) calloc(n * n, sizeof(float));
    assert(h_G);

    /* allocate device memory */
	cudaError_t err = cudaMalloc((void **) &d_G, size);
    cudaMemset(d_G, 0, size); // Initialize the device memory to zero
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector G (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
	//cudaMemcpy(d_G, h_G, size, cudaMemcpyHostToDevice);

    curandState *d_states;

    cudaError_t cudaStatus;
    /* Initialize cuRAND states */
    cudaStatus = cudaMalloc((void **) &d_states, total_threads * sizeof(curandState));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for cuRAND states: %s\n", cudaGetErrorString(cudaStatus));
        return 1; // or exit(EXIT_FAILURE);
    }
    
    //printf("cudaMalloc status d_states: %s\n", cudaGetErrorString(cudaGetLastError()));
    initCurandStates<<<nblocks, ntpb>>>(d_states, total_threads);
    cudaDeviceSynchronize(); // Ensure setup_states kernel finishes before proceeding
    //printf("synchronize status States: %s\n", cudaGetErrorString(cudaGetLastError())); 
    

    /* CUDA timers */
    cudaEvent_t start_device, stop_device;  
    float elapsed_time;
    double time_device;

    /* creates CUDA timers but does not start yet */
    cudaEventCreate(&start_device);
    cudaEventCreate(&stop_device);


    //printf("launching kernel with %d blocks, %d threads per block\n", nblocks, ntpb);
    cudaEventRecord( start_device, 0 );  

  	ray_tracing<<<nblocks,ntpb>>>(d_G, d_states, d_samples, N_rays, n, total_threads, L_x, L_y, L_z, C_y);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch ray_tracing kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    //printf("kernel: %s\n", cudaGetErrorString(cudaGetLastError())); 
    
    
    cudaEventRecord( stop_device, 0 );
    cudaEventSynchronize( stop_device );
    cudaEventElapsedTime( &elapsed_time, start_device, stop_device );
    time_device = static_cast<double>(elapsed_time); // Convert to double
    printf("kernel execution time: %f(s)\n",  time_device/1000.);
 

    /* copy data back to memory */
    cudaMemcpy(h_G,d_G,n*n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&samples,d_samples,sizeof(int), cudaMemcpyDeviceToHost);

    //printf("cudaMemcpy status: %s\n", cudaGetErrorString(cudaGetLastError()));

    printf("num samples: %d\n",  samples);
    
    /* Save results to a file */
    char *filename = "double_sphere.bin";
    write_to_file(h_G, n, filename);

    free(h_G);
    cudaFree(d_G);
    cudaFree(d_samples);
    cudaFree(d_states);
    cudaEventDestroy( start_device );
    cudaEventDestroy( stop_device );

    // Stop the clock and calculate the total execution time
    end_total = clock();
    double total_time = (double)(end_total - start_total) / CLOCKS_PER_SEC;
    printf("total execution time: %f(s)\n", total_time);

    return 0;

}
