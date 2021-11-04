#include "../include/graph.cuh"

// host function for parallel bellman ford routine
long long SSSP_GPU(int numVertices, int numEdges, int *csrOffsets, int *csrCords, int *csrWeights, 
              int *distances, int source=0) {
    // initialize the distances array
    for(int i=0; i<numVertices; i++) distances[i] = INT_MAX>>1;
    distances[source] = 0;

    // launch config
    const int numThreads = 1024;
    const int numBlocksV = (numVertices+numThreads-1)/numThreads;
    // const numBlocksE = (numEdges+numOfThreads-1)/numThreads;

    // pointers for arrays on GPU
    int *csrOffsets_g, *csrCords_g, *csrWeights_g;
    int *distances_g;

    // allocate memory on GPU
    cudaMalloc(&csrOffsets_g, sizeof(int)*(numVertices+1));
    cudaMalloc(&csrCords_g, sizeof(int)*(numEdges));
    cudaMalloc(&csrWeights_g, sizeof(int)*(numEdges));
    cudaMalloc(&distances_g, sizeof(int)*numVertices);

    // copy to GPU
    cudaMemcpy(csrOffsets_g, csrOffsets, sizeof(int)*(numVertices+1), cudaMemcpyHostToDevice);
    cudaMemcpy(csrCords_g, csrCords, sizeof(int)*(numEdges), cudaMemcpyHostToDevice);
    cudaMemcpy(csrWeights_g, csrWeights, sizeof(int)*(numEdges), cudaMemcpyHostToDevice);
    cudaMemcpy(distances_g, distances, sizeof(int)*(numVertices), cudaMemcpyHostToDevice);

    // call kernel to compute edge relaxing numVertices-1 times
    for(int i=0; i<numVertices-1; i++) {
      sssp_kernel<<<numBlocksV, numThreads>>>(csrOffsets_g, csrCords_g, csrWeights_g, distances_g, numVertices, numEdges);
      cudaDeviceSynchronize();

      // check for error
      cudaError_t error = cudaGetLastError();
      if(error != cudaSuccess) {
        // print the CUDA error message
        printf("CUDA error: %s\n", cudaGetErrorString(error));
      }
    }

    // copy distances back to CPU
    cudaMemcpy(distances, distances_g, sizeof(int)*(numVertices), cudaMemcpyDeviceToHost);
    
    /////////////////////////////////////////////////////////////////////////////////
    // for(int i=0; i<=numVertices; i++) printf("%d ", csrOffsets[i]);
    // printf("\n");
    // for(int i=0; i<numEdges; i++) printf("%d ", csrCords[i]);
    // printf("\n");
    // for(int i=0; i<numEdges; i++) printf("%d ", csrWeights[i]);
    // printf("\n");
    /////////////////////////////////////////////////////////////////////////////////

    long long sum = 0;
    for(int i=0; i<numVertices; i++) sum += distances[i];
    return sum;
}