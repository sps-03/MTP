#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <utility>
#include <stdio.h>
#include <limits.h>
#include <cuda.h>

// structure for storing edge information
struct edgeInfo {
    int src, dest, weight;
};

// HOST FUNCTIONS - DECLARATION:
bool compareTwoEdges(const edgeInfo &a, const edgeInfo &b);
long long SSSP_GPU(int numVertices, int numEdges, int *csrOffsets, int *csrCords, int *csrWeights, int *distances, int source);
long long dijkstra(int numVertices, const std::vector<std::vector<int>> &adjList, const std::map<std::pair<int,int>, int> &weights, int *distances, int source);
bool checkDistances(int *distance_1, int *distance_2, int numVertices);
void printDistances(int *distances, int numVertices);

// DEVICE FUNCTIONS - DECLARATION:
template <typename T>
__global__ void init_kernel(T *array, T val, int arraySize);
__global__ void sssp_kernel(int *csrOffsets_g, int *csrCords_g, int *csrWeights_g, int *distances_g, int numVertices, bool *modified_g, bool *modified_next_g, bool *finished_g);

// main function
int main(int argc, char **argv) {
    // if input or output file name is not provided then exit
    if(argc != 3) {
        printf("Enter the input and output file path in the command line.\n");
        return 0;
    }

    // open files
    char *inputFile = argv[1];
    char *outputFile = argv[2];
    FILE *inputFilePtr = fopen(inputFile, "r");
    FILE *outputFilePtr = fopen(outputFile, "w");

    // if not able to open the input file then exit
    if(inputFilePtr == NULL) {
        printf("Failed to open the input file.\n");
        return 0;
    }

    // declaration of variables
    int numVertices, numEdges, startVertex;
    fscanf(inputFilePtr, "%d", &numVertices);
    fscanf(inputFilePtr, "%d", &numEdges);
    fscanf(inputFilePtr, "%d", &startVertex);

    // to store the input graph in COO format
    std::vector<edgeInfo> COO(numEdges); // for directed graphs
    // std::vector<edgeInfo> COO(2*numEdges); // for undirected graphs

    // data structures for storing the graph (as adjacency list) and edge weights
    std::vector<std::vector<int>> adjList(numVertices);
    std::map<std::pair<int,int>, int> weights;

    // read from the input file and populate the COO
    for(int i=0; i<numEdges; i++) {
        int src, dest, weight;
        fscanf(inputFilePtr, "%d %d %d", &src, &dest, &weight);
        // fscanf(inputFilePtr, "%d %d", &src, &dest); // soc-liveJournel
        // weight = 1; // soc-liveJournel

        COO[i].src = src;
        COO[i].dest = dest;
        COO[i].weight = weight;

        adjList[src].push_back(dest);
        weights[{src,dest}] = weight;

        // // for undirected graphs
        // COO[numEdges+i].src = dest;
        // COO[numEdges+i].dest = src;
        // COO[numEdges+i].weight = weight;

        // adjList[dest].push_back(src);
        // weights[{dest,src}] = weight;
    }
    // numEdges = 2*numEdges; // for undirected graph

    // sort the COO 
    std::sort(COO.begin(), COO.end(), compareTwoEdges);

    // converting the graph in COO format to CSR format
    int *csrOffsets = (int*)malloc(sizeof(int)*(numVertices+1));
    int *csrCords = (int*)malloc(sizeof(int)*(numEdges));
    int *csrWeights = (int*)malloc(sizeof(int)*(numEdges));

    // initialize the Offsets array
    for(int i=0; i<numVertices+1; i++) csrOffsets[i] = 0;

    // update the Coordinates and Weights array
    for(int i=0; i<numEdges; i++) {
        csrCords[i] = COO[i].dest;
        csrWeights[i] = COO[i].weight;
    }

    // update the Offsets array
    for(int i=0; i<numEdges; i++) csrOffsets[COO[i].src+1]++;		//store the frequency
    for(int i=0; i<numVertices; i++) csrOffsets[i+1] += csrOffsets[i];	// do cumulative sum
    // converting the graph to CSR done

    // shortest distances from start vertex
    // distance_1 is computed using GPU and distance_2 is computed using CPU
    int *distance_1 = (int*)malloc(sizeof(int)*numVertices);
    int *distance_2 = (int*)malloc(sizeof(int)*numVertices);

    // compute the shortest paths
    long long gpuTotalPathSum = SSSP_GPU(numVertices, numEdges, csrOffsets, csrCords, csrWeights, distance_1, startVertex);
    long long cpuTotalPathSum = dijkstra(numVertices, adjList, weights, distance_2, startVertex);

    /////////////////////////////////////////////////////////////////////////////
    // printf("\n----------------------------------------\n");
    // for(int i=0; i<numEdges; i++) printf("%d %d %d\n", COO[i].src, COO[i].dest, COO[i].weight);
    // printf("----------------------------------------\n");
    // for(int i=0; i<=numVertices; i++) printf("%d ", csrOffsets[i]);
    // printf("\n");
    // for(int i=0; i<numEdges; i++) printf("%d ", csrCords[i]);
    // printf("\n");
    // for(int i=0; i<numEdges; i++) printf("%d ", csrWeights[i]);
    // printf("\n----------------------------------------\n");
    // printf("SSSP_GPU: ");
    // for(int i=0; i<numVertices; i++) printf("%d ", distance_1[i]);
    // printf("\n");
    // printf("SSSP_CPU: ");
    // for(int i=0; i<numVertices; i++) printf("%d ", distance_2[i]);
    // printf("\n----------------------------------------\n");
    /////////////////////////////////////////////////////////////////////////////

    // check for path sum
    if(gpuTotalPathSum != cpuTotalPathSum) {
        printf("Difference in CPU & GPU paths.!!!\n");
        return 0;
    }

    // check whether both the distance arrays are same or not
    if(checkDistances(distance_1, distance_2, numVertices) == false) {
        printf("Check failed..!!!\n");
        return 0;
    }

    // print success message
    printf("Computed SSSP successfully.\n");

    // print the shortest path distances
    // printDistances(distance_1, numVertices);

    // write the result to output file
    for(int i=0; i<numVertices; i++) {
        if(distance_1[i]==INT_MAX) {
            fprintf(outputFilePtr, "The distance to vertex %d is INF\n", i);
            continue;
        }
        fprintf(outputFilePtr, "The distance to vertex %d is %d\n", i, distance_1[i]);
    }

    // free memory allocated on CPU
    free(csrOffsets);
    free(csrCords);
    free(csrWeights);
    free(distance_1);
    free(distance_2);

    // close files
    fclose(inputFilePtr);
    fclose(outputFilePtr);

    return 0;
}

// comparator function
bool compareTwoEdges(const edgeInfo &a, const edgeInfo &b) {
    if(a.src != b.src) return a.src < b.src;
    return a.dest < b.dest;
}

// host function to check whether two distance arrays are same of not
bool checkDistances(int *distance_1, int *distance_2, int numVertices) {
    for(int i=0; i<numVertices; i++) {
        if(distance_1[i] != distance_2[i]) return false;
    }
    return true;
}

// host function to print the distance of each vertex
void printDistances(int *distances, int numVertices) {
    for(int i=0; i<numVertices; i++) {
        printf("The distance to vertex %d is %d\n", i, distances[i]);
    }
}

// host function to compute shortest path using dijkstra's algorithm
long long dijkstra(int numVertices, const std::vector<std::vector<int>> &adjList, const std::map<std::pair<int,int>, int> &weights, int *distances, int source=0) {
    for(int i=0; i<numVertices; i++) distances[i] = INT_MAX;
    distances[source] = 0;

    std::set<std::pair<int,int>> active_vertices;
    active_vertices.insert({0, source});

    while(!active_vertices.empty()) {
        int u = active_vertices.begin()->second;
        active_vertices.erase(active_vertices.begin());
        for(int v : adjList[u]) {
            int newdist = distances[u] + weights.at({u,v});
            if(newdist < distances[v]) {
                active_vertices.erase({distances[v], v});
                distances[v] = newdist;
                active_vertices.insert({newdist, v});
            }
        }
    }

    long long sum = 0;
    for(int i=0; i<numVertices; i++) sum += distances[i];
    return sum;
}

// host function for parallel bellman ford (modified) routine
long long SSSP_GPU(int numVertices, int numEdges, int *csrOffsets, int *csrCords, int *csrWeights, int *distances, int source=0) {
    // launch config
    const int numThreads = 1024;
    const int numBlocksV = (numVertices+numThreads-1)/numThreads;
    // const numBlocksE = (numEdges+numOfThreads-1)/numThreads;

    // pointers for arrays on GPU
    int *csrOffsets_g, *csrCords_g, *csrWeights_g;
    int *distances_g;
    bool *modified_g, *modified_next_g, *finished_g;

    // allocate memory on GPU
    cudaMalloc(&csrOffsets_g, sizeof(int)*(numVertices+1));
    cudaMalloc(&csrCords_g, sizeof(int)*(numEdges));
    cudaMalloc(&csrWeights_g, sizeof(int)*(numEdges));
    cudaMalloc(&distances_g, sizeof(int)*numVertices);
    cudaMalloc(&modified_g, sizeof(bool)*numVertices);
    cudaMalloc(&modified_next_g, sizeof(bool)*numVertices);
    cudaMalloc(&finished_g, sizeof(bool));

    // pointers other arrays on GPU
    bool *modified = (bool*)malloc(sizeof(bool)*numVertices);
    bool *finished = (bool*)malloc(sizeof(bool));

    // initialize the CPU arrays
    for(int i=0; i<numVertices; i++) {
        distances[i] = INT_MAX;
        modified[i] = false;
    }
    distances[source] = 0;
    modified[source] = true;
    *finished = false;

    // for recording the total time taken
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // copy to GPU
    cudaMemcpy(csrOffsets_g, csrOffsets, sizeof(int)*(numVertices+1), cudaMemcpyHostToDevice);
    cudaMemcpy(csrCords_g, csrCords, sizeof(int)*(numEdges), cudaMemcpyHostToDevice);
    cudaMemcpy(csrWeights_g, csrWeights, sizeof(int)*(numEdges), cudaMemcpyHostToDevice);
    cudaMemcpy(distances_g, distances, sizeof(int)*(numVertices), cudaMemcpyHostToDevice);
    cudaMemcpy(modified_g, modified, sizeof(bool)*(numVertices), cudaMemcpyHostToDevice);
    cudaMemcpy(finished_g, finished, sizeof(bool), cudaMemcpyHostToDevice);

    // call kernel to compute edge relaxing till no more updates or at max "numVertices-1" times
    int iter = 0;
    init_kernel<bool><<<numBlocksV, numThreads>>>(modified_next_g, false, numVertices);
    while(*finished != true) {
        *finished = true;
        init_kernel<bool><<<1, 1>>>(finished_g, true, 1);
        sssp_kernel<<<numBlocksV, numThreads>>>(csrOffsets_g, csrCords_g, csrWeights_g, distances_g, numVertices, modified_g, modified_next_g, finished_g);
        init_kernel<bool><<<numBlocksV, numThreads>>>(modified_g, false, numVertices);
        cudaDeviceSynchronize();

        // // check for error
        // cudaError_t error = cudaGetLastError();
        // if(error != cudaSuccess) {
        //     // print the CUDA error message
        //     printf("CUDA error: %s\n", cudaGetErrorString(error));
        // }

        cudaMemcpy(finished, finished_g, sizeof(bool), cudaMemcpyDeviceToHost);
        bool *tempPtr = modified_next_g;
        modified_next_g = modified_g;
        modified_g = tempPtr;

        if(++iter == numVertices-1) break;
    }
    
    // copy distances back to CPU
    cudaMemcpy(distances, distances_g, sizeof(int)*(numVertices), cudaMemcpyDeviceToHost);

    // print time taken
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time Taken: %.6f ms \nIterations: %d\n", milliseconds, iter);

    // free up the memory
    free(modified);
    free(finished);
    cudaFree(csrOffsets_g);
    cudaFree(csrCords_g);
    cudaFree(csrWeights_g);
    cudaFree(distances_g);
    cudaFree(modified_g);
    cudaFree(modified_next_g);
    cudaFree(finished_g);

    long long sum = 0;
    for(int i=0; i<numVertices; i++) sum += distances[i];
    return sum;
}

// kernel invoked in parallel bellman ford sssp routine
__global__ void sssp_kernel(int *csrOffsets_g, int *csrCords_g, int *csrWeights_g, int *distances_g, int numVertices, bool *modified_g, bool *modified_next_g, bool *finished_g) {
    unsigned int id = blockDim.x*blockIdx.x + threadIdx.x;
    if(id < numVertices) {
        if(modified_g[id]) {
            for(int e=csrOffsets_g[id]; e<csrOffsets_g[id+1]; e++) {
                int v = csrCords_g[e];
                int new_dist = distances_g[id] + csrWeights_g[e];
                if(new_dist < distances_g[v]) {
                    atomicMin(&distances_g[v] , new_dist);
                    modified_next_g[v] = true;
                    *finished_g = false;
                }
            }
        }
    }
}

// kernel for value initialization
template <typename T>
__global__ void init_kernel(T *array, T val, int arraySize) {
    unsigned int id = blockDim.x*blockIdx.x + threadIdx.x;
    if(id < arraySize) array[id] = val;
}