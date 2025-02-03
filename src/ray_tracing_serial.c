#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#define PI 3.14159265

 // Predefined parameters
double L_x = 4;					// light source x
double L_y = 4;					// light source y
double L_z = -1;			    // light source z
double W_y = 2;				    // window y
double W_max = 2;				// window width
double C_y = 12;				// sphere y  
double R = 6;
int n = 1024;

// Undefined parameters
double *G;
double t; 
double b; 
double temp;
int row, col;
int i_index, j_index;



void initialize_grid(int n){
    G = (double *)calloc(n * n, sizeof(double));
}

typedef struct vectData {
	double x;
	double y;
	double z;
} vector;

vector create_vector(double x, double y, double z)
{
	vector vec;
	vec.x = x;
	vec.y = y;
	vec.z = z;
	return vec;
}

double dot_product(vector A, vector B)
{
    double C = (A.x * B.x) + (A.y * B.y) + (A.z * B.z);
    return C;
}

// Function to generate a random double between min and max
double randDouble(double min, double max) {
    return min + (rand() / (RAND_MAX / (max - min)));
}

// Helper function to set vector V 
vector sample_rand(vector A){
    double phi = randDouble(0, 2 * PI);
    double cosTheta = randDouble(-1, 1);
    double sinTheta = sqrt(1 - cosTheta * cosTheta);

    A.x = sinTheta * cos(phi);
    A.y = sinTheta * sin(phi);
    A.z = cosTheta;
    return A;
}

// Helper function for t
double calculate(vector V, vector C, double R){
    double vc = dot_product(V, C);
    double cc = dot_product(C, C);
    double calc = pow(vc, 2) + pow(R, 2) - cc;
    return calc;
}


/* Vector math functions */

vector difference(vector A, vector B)
{
    vector C;
    C.x = A.x - B.x;
	C.y = A.y - B.y;
	C.z = A.z - B.z;

	return C;
}

vector divide(vector V, double t){
    V.x = V.x / t;
    V.y = V.y / t;
    V.z = V.z / t;
    return V;
}

vector intersection(double t, vector V){
    vector I;
    I.x = t * V.x;
    I.y = t * V.y;
    I.z = t * V.z;
    return I;
}

vector unit_normal(vector I, vector C){
    vector vect, N;
    vect = difference(I, C);
    double d = sqrt(pow(vect.x, 2) + pow(vect.y, 2) + pow(vect.z, 2));
    N = divide(vect, d);
    return N;
}

vector direction(vector L, vector I){
    vector vect, S;
    vect = difference(L, I);
    double d = sqrt(pow(vect.x, 2) + pow(vect.y, 2) + pow(vect.z, 2));
    S = divide(vect, d);
    return S;
}

double max(double a, double b)
{
	if (a >= b)
		return a;
	else
		return b;
}

// Write G data to file
void write_to_file(double *G, int n, char *file_name)
{
	FILE *fp = fopen(file_name, "w");

	for (int i=(n-1); i>=0; i--)
	{
		for(int j=0; j<n; j++)
		 	fprintf(fp, "%f ", G[j + (n * i)]);
		fprintf(fp, "\n");
	}
	fclose(fp);
}

int main(int argc, char * argv[])
{
    int N_rays = atoi(argv[1]);
    //int n = atoi(argv[2]);

    //int N_rays = 1000000000;
    //int n = 1024;

    srand(time(NULL));
    //srand(44);
    initialize_grid(n);

    char *filename = "sphere.txt";

    vector V;
    vector I;
    vector N;
    vector S;

    vector L = create_vector(L_x, L_y, L_z);	
    vector C = create_vector(0, C_y, 0);	
    vector W = create_vector(0, W_y, 0);	


    // Start time
    double runtime = omp_get_wtime(); 
    int nt = 1;
    omp_set_num_threads(nt);


    #pragma omp parallel for default(none) shared(stdout, N_rays, n, G, R, C, L, W_max, W, t, I, N, S, b) \
                                            private(temp, V, i_index, j_index) 

    for(int ray=0; ray<N_rays; ray++){
        temp = -1;
        while((fabs(W.x) > W_max) || (fabs(W.z) > W_max) || temp <= 0 ){
            //printf("W.y: %f, V.y:%f\n",W.y, V.y);
            V = sample_rand(V);
            if(V.y!=0.0){
                W = intersection((W.y / V.y), V);
                temp = calculate(V, C, R);
            }
            
        }
        
        t = dot_product(V, C) - sqrt(temp);
        I = intersection(t, V);
        N = unit_normal(I, C);
        S = direction(L, I);

        b = max(0, dot_product(S, N));

        int i_index = round(n * (W.x + W_max) / (2*W_max));
	    int j_index = round(n * (W.z + W_max) / (2*W_max));

        if(fabs(b)>1000){
            printf("Very large b\n");
        }


        G[i_index * n + j_index] += b;


    }

    runtime = omp_get_wtime() - runtime;
    printf("time(s): %f\n", runtime);

    printf("Writing G\n");
    write_to_file(G, n, filename);
    free(G);
    return 0;
}