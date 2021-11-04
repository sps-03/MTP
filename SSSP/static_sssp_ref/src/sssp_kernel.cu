#include "../include/graph.cuh"

// kernel invoked in parallel bellman ford sssp routine
__global__ void sssp_kernel(int *csrOffsets_g, int *csrCords_g, int *csrWeights_g, 
                            int *distances_g, int numVertices, int numEdges) {
    unsigned int id = blockDim.x*blockIdx.x + threadIdx.x;
    if(id < numVertices) {
        int start = csrOffsets_g[id];
        int end = csrOffsets_g[id+1];
        int u = id;
        for(int i=start; i<end; i++) {
            int v = csrCords_g[i];
            int w = csrWeights_g[i];
            atomicMin(&distances_g[v], distances_g[u]+w);
        }
    }
}