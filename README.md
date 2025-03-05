# RayTracing_HPC
Ray Tracing on a Sphere: CUDA Performance Analysis

## Introduction
This report outlines a GPU Ray Tracing algorithm with CUDA and compares its execution
time with a serial version, parallel implementation, and across different GPUs using
CUDA. The goal of the programs is to render a 3D reflective sphere illuminated by a 
single light source.



Parallel Implementation
To improve the performance of the serial implementation, OpenMP was implemented to
parallelize the code and distribute work for the ray tracing function for a multi-core CPU. To
ensure thread safety, the vector computations were performed locally, without the use of helper
functions. This modification also increased the runtime further as the functions did not have to be
called at every step. The rand_r() function was used to ensure thread safety across computations
and updates.
## Implementation
### Initialization
For each implementation, the functions were run on a 1000 x 1000 grid, for 1 billion light rays. The 1000 x 1000 grid
 was initialized with zeros and was iteratively updated at every calculation.

#### Serial
The serial implementation for this was used as a baseline benchmark to measure the performance of parallelization 
techniques. Here, the ray tracing algorithm processes each pixel one at a time. For
each pixel, it calculates the light's path, including reflections, refractions, and shadows, to
determine the final color of that pixel. The algorithm iterates over all pixels in the image,
rendering each scene sequentially. This approach, while straightforward, does not
leverage modern multi-core processors' capabilities, leading to long rendering times for
complex scenes or high-resolution images.  For the serial implementation, I utilized helper functions 
and the rand() method as they would not be impacted by threading. Without any parallelization, 1 billion rays
 took 5.5 minutes to run on the midway server. 


**Usage:**

\$ gcc -fopenmp -O3 -g -o serial ray_tracing_serial.c -lm

\$ ./serial

#### Parallel
The serial implementation was further enhanced through implementing OpenMP and parallelizing the ray tracing function
 for a multi-core CPU.  To
ensure thread safety, the vector computations were performed locally, without the use of helper
functions. This modification also increased the runtime further as the functions did not have to be
called at every step. The rand_r() function was used to ensure thread safety across computations.

On 3 threads, the program had an execution time of 2.26 minutes and a speedup of 2.56 when compared to the serial implementation,
 and on 16 threads the time was 26.5 seconds with a speedup of 13.16 compared to the serial implementation.

**Usage:**

\$ gcc -fopenmp -O3 -g -o parallel ray_tracing_parallel.c -lm ./parallel 3

\$ ./parallel 16

#### CUDA: GPU Multi threading
To further optimize the computation time, CUDA was invoked and compared across a
variety of available GPUs. The CUDA implementation parallelizes the ray tracing
process by assigning the computation of each pixel (or a small group of pixels) to
separate threads on the GPU. This allows for the simultaneous calculation of multiple
pixels, drastically reducing the total rendering time. The CUDA model divides the image
into blocks and threads, where each thread calculates the color for a single pixel, and
each block contains a group of threads working together. The host-side code is responsible
for initialization of the final grid, setup tasks, and memory allocation on the GPU. It also
ensures that data is safely transferred between the host and the device. The results are then
 gathered back on the host for final image composition and output on the grid. For consistency, all
computations were performed on 1000 blocks. The number of threads per block varied
according to GPU availability and optimal performance. This approach takes full
advantage of the GPU's architecture, designed for high-throughput parallel computations.

For the computation of the randomized vector, I used CUDA’s cuRAND function to ensure thread-safety. With cuRAND, each thread 
can independently sample the directions for the vector which will not lead to conflicts during parallelization.

The CUDA solution was incredibly efficient with a completion time of 0.459s and a speedup of 758.32 times compared to the serial
 implementation.

## Performance Comparison
To compare the performance of the serial and CUDA implementations, we measure the
rendering time of a sphere with varying GPUs, grid dimensions, thread number (per
block), and single vs. double precision. The performance metrics are measured in the
total time the program took to execute the ray tracing function and the total program
computation time. These standardizations ensured a fair comparison between the CPU
and GPU processing capabilities.

## Optimization Strategies
### Helper/Built-in Functions:
Throughout this project, I observed a variety of techniques that reduced the computation
time. One of the best optimizations I observed was bringing the helper functions I had into the
driver function. This included removing built-in math functions, such as pow(), and calculating
everything manually. Having these computations done locally reduced the running time by 0.4s.
Surprisingly, I found that the fabs() function performed 0.2s better than the manual
computation in single precision, but this had a worse performance in double precision.

### Writing to File:
Writing data to a binary file rather than a CSV file often had a shorter runtime compared
to writing to a text file. This was most likely because binary files are a more efficient data
representation and result in a reduced file size. Because we were generating and storing large
amounts of data, this reflected in the total runtime and performance of the program.

### Double vs. Single Precision:
The precision of computation, single or double, significantly affected both performance
and memory usage. Single precision using 32-bit floating-point numbers had better execution
speeds compared to double precision using 64-bit floating-point numbers. This makes sense
because single-precision uses less memory in the program.

### Different Flags:
Flags such as -use_fast_math increased the speed by 0.1s, which was not too big but still
something interesting.

### Possible further optimizations:
Integrating MPI with CUDA for distributed parallelism would further optimize this
program. This would scale CUDA beyond the GPU resources of a single node,
facilitating distributed parallelism across multiple GPUs and nodes.
 
**Usage:**

\$ nvcc -O3 -use_fast_math -o ray_tracings ray_tracing_cuda.cu -arch=sm_75 ./ray_tracings

## Data Visualization and Performances
The computed results of the n x n grid were stored in a file and visualized with python code. They simulared the visualization 
of the sphere when all the rays have been computed. The states were plotted using matplotlib.

## Conclusion
The time comparisons between a serial implementation, parallelization with OpemMP, and parallelization over GPUs with CUDA. In
 every implementation, the sphere was correctly cons- tructed, but with various parallelization techniques, the execution time 
 differed greatly. The CUDA implementation on GPUs had the best and fastest performance. For future, use this performance could 
 be further optimized to perform even better.
