#ifndef __GRAPH__HEADER__CUDA__
#define __GRAPH__HEADER__CUDA__

#include <cuda.h>
#include "graph.h"

// function prototypes for kernels
__global__ void init_distance_kernel(int *distances, int source, int numVertices);
__global__ void sssp_kernel(int *csrOffsets_g, int *csrCords_g, int *csrWeights_g, 
                            int *distances_g, int numVertices, int numEdges);

#endif