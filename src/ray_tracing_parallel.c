#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
//#include<cuda.h>
#include <assert.h>
#define PI 3.14159265


// Predefined parameters
long int N_rays = 1000000000;
double W_y = 2.0;
double W_max = 2.0;
double C_y = 12.0;
double R = 6.0;
double L_x = 4.0;
double L_y = 4.0;
double L_z = -1.0;
int n = 1000;

// Undefined parameters
double *G;

void initialize_grid(int n){
    G = (double *)calloc(n * n, sizeof(double));
}

typedef struct vectData {
	double x, y, z;
} vector;


double max(double a, double b){
	if (a >= b)
		return a;
	else
		return b;
}

// Write G data to file
void write_to_file(double *G, int n, char *file_name){
	FILE *fp = fopen(file_name, "w");

	for (int i=(n-1); i>=0; i--){
		for(int j=0; j<n; j++)
		 	fprintf(fp, "%f ", G[j + (n * i)]);
		fprintf(fp, "\n");
	}
	fclose(fp);
}

void ray_tracing() {
    vector W,C,L;

    W.x = 0.0;
    W.y = W_y;
    W.z = 0.0;

    L.x = L_x;
    L.y = L_y;
    L.z = L_z;

    C.x = 0.0;
    C.y = C_y;
    C.z = 0.0;
	
    initialize_grid(n);

    #pragma omp parallel shared(G)
    {
        unsigned int seed = time(NULL) ^ omp_get_thread_num(); // Unique seed per thread;
        #pragma omp for private(W)
        for(int i=0; i<N_rays;i++){ 
            vector V,I,N,S;

            W.x = 0.0;
            W.y = W_y;
            W.z = 0.0;

            double temp = -1;
            while((fabs(W.x) > W_max) || (fabs(W.z) > W_max) || temp <= 0 ){ 

                double phi = ((double)rand_r(&seed) / RAND_MAX) * PI;
                double cos_theta = 2.0 * ((double)rand_r(&seed) / RAND_MAX) - 1.0;
                double sin_theta = sqrt(1.0 - (cos_theta * cos_theta));

                V.x = sin_theta * cos(phi);
                V.y = sin_theta * sin(phi);
                V.z = cos_theta;


                if (V.y != 0.0) {
                    //printf("temp %f, V.x%f, V.y%f, V.z%f, W_max%f\n", temp, V.x, V.y, V.z, W_max);
                    W.x = (W.y / V.y) * V.x;
                    W.z = (W.y / V.y) * V.z;

                    double vc = (V.x * C.x) + (V.y * C.y) + (V.z * C.z);
                    double cc = (C.x * C.x) + (C.y * C.y) + (C.z * C.z);
                    temp = (vc * vc) + (R * R) - cc;
                }
            }
            
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

            double b = max(0.0, ((S.x * N.x) + (S.y * N.y) + (S.z * N.z)));

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

            #pragma omp atomic
            G[i_index * n + j_index] += b;            
        }
    }
}


int main(int argc, char * argv[]){

    int num_threads = atoi(argv[1]);
    omp_set_num_threads(num_threads); 
    char *filename = "parallel_sphere_3.txt";

    // Start time
    double runtime = omp_get_wtime(); 

    ray_tracing();

    runtime = omp_get_wtime() - runtime;
    printf("time(s): %f\n", runtime);

    printf("Writing G\n");
    write_to_file(G, n, filename);
    free(G);
    return 0;
}