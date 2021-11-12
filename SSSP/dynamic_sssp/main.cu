#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <utility>
#include <stdio.h>
#include <cuda.h>
#define MAX_INT 2147483647

// structure for storing edge information
struct edgeInfo {
    int src, dest, weight;
};

// HOST FUNCTIONS - DECLARATION:
bool compareTwoEdges(const edgeInfo &a, const edgeInfo &b);
bool compareTwoEdgesR(const edgeInfo &a, const edgeInfo &b);
long long SSSP_GPU(int numVertices, int numEdges, int *csrOffsets, int *csrCords, int *csrWeights, 
                   int *distances, int *parent, int source);
long long dijkstra(int numVertices, const std::vector<std::vector<int>> &adjList, 
                   const std::map<std::pair<int,int>, int> &edgeWeights, int *distances, int source);
bool checkDistances(int *distances1, int *distances2, int numVertices);
void printArray(int *arr, int len);
int find(int dest, int *csrCords, int start, int end);
void addEdge(int u, int v, int w, int &numVertices, int &numEdges, int *distances, int *parent, 
             int *&csrOffsets, int *&csrCords, int *&csrWeights, 
             int *&csrOffsetsR, int *&csrCordsR, int *&csrWeightsR, 
             std::vector<std::vector<int>> &adjList, std::map<std::pair<int,int>, int> &edgeWeights);
void deleteEdge(int u, int v, int &numVertices, int &numEdges, int *distances, int *parent, 
                int *&csrOffsets, int *&csrCords, int *&csrWeights, 
                int *&csrOffsetsR, int *&csrCordsR, int *&csrWeightsR, 
                std::vector<std::vector<int>> &adjList, std::map<std::pair<int,int>, int> &edgeWeights);


// DEVICE FUNCTIONS - DECLARATION:
template <typename T>
__global__ void init_kernel(T *array, T val, int arraySize);
__global__ void sssp_kernel(int *csrOffsets_d, int *csrCords_d, int *csrWeights_d, int *distances_d, 
                            int *parent_d, int *locks_d, int numVertices, bool *modified_d, 
                            bool *modified_next_d, bool *finished_d);
__global__ void mark_descendants(int *distances_d, int *parent_d, bool *modified_d, 
                                 int numVertices, bool *finished_d);
__global__ void fetch_and_update(int *csrOffsetsR_d, int *csrCordsR_d, int *csrWeightsR_d, int *distances_d, 
                                 int *parent_d, int numVertices, bool *modified_d, bool *finished_d);
__global__ void set_unreachable(int *distances_d, int *parent_d, int numVertices);


// main function
int main(int argc, char **argv) {
    // if input, update or output file names are not provided then exit
    if(argc != 4) {
        printf("Enter the input, update and output file path in the command line.\n");
        return 0;
    }

    // read file names
    char *inputFile = argv[1];
    char *updateFile = argv[2];
    char *outputFile = argv[3];

    // open input file
    FILE *inputFilePtr = fopen(inputFile, "r");
    FILE *updateFilePtr = fopen(updateFile, "r");
    
    // if not able to open the input file then exit
    if(inputFilePtr == NULL) {
        printf("Failed to open the input file.\n");
        return 0;
    }

    // if not able to open the update file then exit
    if(updateFilePtr == NULL) {
        printf("Failed to open the update file.\n");
        return 0;
    }

    // declaration of variables
    int numVertices, numEdges, startVertex;
    fscanf(inputFilePtr, "%d", &numVertices);
    fscanf(inputFilePtr, "%d", &numEdges);
    fscanf(inputFilePtr, "%d", &startVertex);

    // to store the input graph in COO format
    std::vector<edgeInfo> COO(numEdges);        ///// for directed graphs
    // std::vector<edgeInfo> COO(2*numEdges);   ///// for undirected graphs

    // data structures for storing the graph (as adjacency list) and edge weights
    // used while computing SSSP on CPU
    std::vector<std::vector<int>> adjList(numVertices);
    std::map<std::pair<int,int>, int> edgeWeights;

    // read from the input file and populate the COO
    for(int i=0; i<numEdges; i++) {
        int src, dest, weight;
        fscanf(inputFilePtr, "%d %d %d", &src, &dest, &weight);
        // fscanf(inputFilePtr, "%d %d", &src, &dest);  ///// soc-liveJournel
        // weight = 1;                                  ///// soc-liveJournel

        COO[i].src = src;
        COO[i].dest = dest;
        COO[i].weight = weight;
        adjList[src].push_back(dest);
        edgeWeights[{src,dest}] = weight;

        // // for undirected graphs
        // COO[numEdges+i].src = dest;
        // COO[numEdges+i].dest = src;
        // COO[numEdges+i].weight = weight;
        // adjList[dest].push_back(src);
        // edgeWeights[{dest,src}] = weight;
    }
    // numEdges = 2*numEdges;                   // for undirected graph

    // close input file
    fclose(inputFilePtr);

    // sort the COO
    std::sort(COO.begin(), COO.end(), compareTwoEdges);

    // converting the graph in COO format to CSR format
    int *csrOffsets = (int*)malloc(sizeof(int)*(numVertices+1));
    int *csrCords = (int*)malloc(sizeof(int)*(numEdges));
    int *csrWeights = (int*)malloc(sizeof(int)*(numEdges));

    // initialize the Offsets array
    for(int i=0; i<=numVertices; i++) csrOffsets[i] = 0;

    // update the Coordinates and Weights array
    for(int i=0; i<numEdges; i++) {
        csrCords[i] = COO[i].dest;
        csrWeights[i] = COO[i].weight;
    }

    // update the Offsets array
    for(int i=0; i<numEdges; i++) csrOffsets[COO[i].src+1]++;		// store the frequency
    for(int i=0; i<numVertices; i++) csrOffsets[i+1] += csrOffsets[i];	// do cumulative sum

    // sort the COO (for reverseCSR)
    std::sort(COO.begin(), COO.end(), compareTwoEdgesR);

    // converting the graph in COO format to reverseCSR format
    int *csrOffsetsR = (int*)malloc(sizeof(int)*(numVertices+1));
    int *csrCordsR = (int*)malloc(sizeof(int)*(numEdges));
    int *csrWeightsR = (int*)malloc(sizeof(int)*(numEdges));

    // initialize the Offsets array
    for(int i=0; i<=numVertices; i++) csrOffsetsR[i] = 0;

    // update the Coordinates and Weights array
    for(int i=0; i<numEdges; i++) {
        csrCordsR[i] = COO[i].src;
        csrWeightsR[i] = COO[i].weight;
    }

    // update the Offsets array
    for(int i=0; i<numEdges; i++) csrOffsetsR[COO[i].dest+1]++;		// store the frequency
    for(int i=0; i<numVertices; i++) csrOffsetsR[i+1] += csrOffsetsR[i];	// do cumulative sum
    
    // converting the graph to CSRs done

    // shortest distances from start vertex
    // distances1 is computed using GPU and distances2 is computed using CPU
    int *distances1 = (int*)malloc(sizeof(int)*numVertices);
    int *distances2 = (int*)malloc(sizeof(int)*numVertices);

    // parent array
    int *parent = (int*)malloc(sizeof(int)*numVertices);

    // compute the shortest paths
    long long gpuTotalPathSum = SSSP_GPU(numVertices, numEdges, csrOffsets, csrCords, csrWeights, 
                                         distances1, parent, startVertex);
    long long cpuTotalPathSum = dijkstra(numVertices, adjList, edgeWeights, distances2, startVertex);

    // check for path sum
    if(gpuTotalPathSum != cpuTotalPathSum) {
        printf("Initial Graph: Difference in CPU & GPU paths.!!!\n");
        return 0;
    }

    // check whether both the distances arrays are same or not
    if(checkDistances(distances1, distances2, numVertices) == false) {
        printf("Initial Graph: Check failed..!!!\n");
        return 0;
    }

    // print success message
    printf("Computed SSSP for initial graph successfully.\n");

    // open output file
    FILE *outputFilePtr = fopen(outputFile, "w");

    // write the result to output file
    fprintf(outputFilePtr, "Distances for the initial graph:\n");
    for(int i=0; i<numVertices; i++) {
        if(distances1[i]==MAX_INT)
            fprintf(outputFilePtr, "The distance to vertex %d is INF\n", i);
        else
            fprintf(outputFilePtr, "The distance to vertex %d is %d\n", i, distances1[i]);
    }
    fprintf(outputFilePtr, "\n");

    
    // start updates:
    int numUpdates, u, v, w;;
    char type;
    fscanf(updateFilePtr, "%d", &numUpdates);
    for(int i=0; i < numUpdates; i++) {
        fprintf(outputFilePtr, "\nUpdate %d : ", i+1);
        printf("\nUpdate %d: ", i+1);
        fscanf(updateFilePtr, " %c", &type);
        if(type=='a') {
            fscanf(updateFilePtr, "%d", &u);
            fscanf(updateFilePtr, "%d", &v);
            fscanf(updateFilePtr, "%d", &w);
            fprintf(outputFilePtr, "%c %d %d %d\n", type, u, v, w);
            printf("%c %d %d %d\n", type, u, v, w);

            addEdge(u, v, w, numVertices, numEdges, distances1, parent, 
                    csrOffsets, csrCords, csrWeights, csrOffsetsR, csrCordsR, csrWeightsR, 
                    adjList, edgeWeights);
        } else if(type=='d') {
            fscanf(updateFilePtr, "%d", &u);
            fscanf(updateFilePtr, "%d", &v);
            fprintf(outputFilePtr, "%c %d %d\n", type, u, v);
            printf("%c %d %d\n", type, u, v);

            deleteEdge(u, v, numVertices, numEdges, distances1, parent, 
                       csrOffsets, csrCords, csrWeights, csrOffsetsR, csrCordsR, csrWeightsR, 
                       adjList, edgeWeights);
        }

        // write the result to output file
        for(int i=0; i<numVertices; i++) {
            if(distances1[i]==MAX_INT)
                fprintf(outputFilePtr, "The distance to vertex %d is INF\n", i);
            else
                fprintf(outputFilePtr, "The distance to vertex %d is %d\n", i, distances1[i]);
        }
        fprintf(outputFilePtr, "\n");
    }
    
    // final checking
    distances2 = (int*)realloc(distances2, sizeof(int)*numVertices);
    cpuTotalPathSum = dijkstra(numVertices, adjList, edgeWeights, distances2, startVertex);
    gpuTotalPathSum = 0; for(int i=0; i<numVertices; i++) gpuTotalPathSum += distances1[i];

    // check for total path sum
    if(gpuTotalPathSum != cpuTotalPathSum) {
        printf("\nFinal: Difference in CPU & GPU paths.!!!\n");
        return 0;
    }

    // check whether both the distances arrays are same or not
    if(checkDistances(distances1, distances2, numVertices) == false) {
        printf("\nFinal: Check failed..!!!\n");
        return 0;
    }

    // print success message
    printf("\nComputed SSSP for final graph successfully.\n");

    // write the final result to output file
    fprintf(outputFilePtr, "Distances for final graph\n");
    for(int i=0; i<numVertices; i++) {
        if(distances1[i]==MAX_INT)
            fprintf(outputFilePtr, "The distance to vertex %d is INF\n", i);
        else
            fprintf(outputFilePtr, "The distance to vertex %d is %d\n", i, distances1[i]);
    }
    fprintf(outputFilePtr, "\n");

    // free memory allocated on CPU
    free(csrOffsets);
    free(csrCords);
    free(csrWeights);
    free(csrOffsetsR);
    free(csrCordsR);
    free(csrWeightsR);
    free(distances1);
    free(distances2);
    free(parent);

    // close files
    fclose(updateFilePtr);
    fclose(outputFilePtr);

    return 0;
}

// comparator function
bool compareTwoEdges(const edgeInfo &a, const edgeInfo &b) {
    if(a.src != b.src) return a.src < b.src;
    return a.dest < b.dest;
}

// comparator function for reverse CSR
bool compareTwoEdgesR(const edgeInfo &a, const edgeInfo &b) {
    if(a.dest != b.dest) return a.dest < b.dest;
    return a.src < b.src;
}

// host function to check whether two distances arrays are same of not
bool checkDistances(int *distances1, int *distances2, int numVertices) {
    for(int i=0; i<numVertices; i++) {
        if(distances1[i] != distances2[i]) return false;
    }
    return true;
}

// host function to print an array content
void printArray(int *arr, int len) {
    for(int i=0; i<len; i++) {
        printf("%d ", arr[i]);
    } printf("\n");
}

// host function to compute shortest path using dijkstra's algorithm
long long dijkstra(int numVertices, const std::vector<std::vector<int>> &adjList, 
                   const std::map<std::pair<int,int>, int> &edgeWeights, int *distances, int source=0) {
    for(int i=0; i<numVertices; i++) distances[i] = MAX_INT;
    distances[source] = 0;

    std::set<std::pair<int,int>> active_vertices;
    active_vertices.insert({0, source});

    while(!active_vertices.empty()) {
        int u = active_vertices.begin()->second;
        active_vertices.erase(active_vertices.begin());
        for(int v : adjList[u]) {
            if(edgeWeights.at({u,v}) == MAX_INT) continue;
            int newDist = distances[u] + edgeWeights.at({u,v});
            if(newDist < distances[v]) {
                active_vertices.erase({distances[v], v});
                distances[v] = newDist;
                active_vertices.insert({newDist, v});
            }
        }
    }

    long long sum = 0;
    for(int i=0; i<numVertices; i++) sum += distances[i];
    return sum;
}

// host function for parallel bellman ford (fixed point) routine
long long SSSP_GPU(int numVertices, int numEdges, int *csrOffsets, int *csrCords, int *csrWeights, 
                   int *distances, int *parent, int source=0) {
    // launch config
    const int numThreads = 1024;
    const int numBlocksV = (numVertices+numThreads-1)/numThreads;
    // const numBlocksE = (numEdges+numOfThreads-1)/numThreads;

    // pointers for arrays on CPU
    bool *modified = (bool*)malloc(sizeof(bool)*numVertices);
    bool *finished = (bool*)malloc(sizeof(bool));
    
    // pointers for arrays on GPU
    int *csrOffsets_d, *csrCords_d, *csrWeights_d;
    int *distances_d, *parent_d, *locks_d;
    bool *modified_d, *modified_next_d, *finished_d;

    // allocate memory on GPU
    cudaMalloc(&csrOffsets_d, sizeof(int)*(numVertices+1));
    cudaMalloc(&csrCords_d, sizeof(int)*(numEdges));
    cudaMalloc(&csrWeights_d, sizeof(int)*(numEdges));
    cudaMalloc(&distances_d, sizeof(int)*numVertices);
    cudaMalloc(&parent_d, sizeof(int)*numVertices);
    cudaMalloc(&locks_d, sizeof(int)*numVertices);
    cudaMalloc(&modified_d, sizeof(bool)*numVertices);
    cudaMalloc(&modified_next_d, sizeof(bool)*numVertices);
    cudaMalloc(&finished_d, sizeof(bool));

    // initialize the CPU arrays
    for(int i=0; i<numVertices; i++) {
        distances[i] = MAX_INT;
        parent[i] = -1;
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
    cudaMemcpy(csrOffsets_d, csrOffsets, sizeof(int)*(numVertices+1), cudaMemcpyHostToDevice);
    cudaMemcpy(csrCords_d, csrCords, sizeof(int)*(numEdges), cudaMemcpyHostToDevice);
    cudaMemcpy(csrWeights_d, csrWeights, sizeof(int)*(numEdges), cudaMemcpyHostToDevice);
    cudaMemcpy(distances_d, distances, sizeof(int)*(numVertices), cudaMemcpyHostToDevice);
    cudaMemcpy(parent_d, parent, sizeof(int)*(numVertices), cudaMemcpyHostToDevice);
    cudaMemcpy(modified_d, modified, sizeof(bool)*(numVertices), cudaMemcpyHostToDevice);

    // call kernel to compute edge relaxing till no more updates or at max "numVertices-1" times
    int iter = 0;
    init_kernel<bool><<<numBlocksV, numThreads>>>(modified_next_d, false, numVertices);
    init_kernel<int><<<numBlocksV, numThreads>>>(locks_d, 0, numVertices);
    while(*finished != true) {
        init_kernel<bool><<<1, 1>>>(finished_d, true, 1);
        sssp_kernel<<<numBlocksV, numThreads>>>(csrOffsets_d, csrCords_d, csrWeights_d, distances_d, parent_d, 
                                                locks_d, numVertices, modified_d, modified_next_d, finished_d);
        init_kernel<bool><<<numBlocksV, numThreads>>>(modified_d, false, numVertices);
        cudaDeviceSynchronize();

        // // check for error
        // cudaError_t error = cudaGetLastError();
        // if(error != cudaSuccess) {
        //     printf("CUDA error: %s\n", cudaGetErrorString(error));
        // }

        cudaMemcpy(finished, finished_d, sizeof(bool), cudaMemcpyDeviceToHost);
        bool *tempPtr = modified_next_d;
        modified_next_d = modified_d;
        modified_d = tempPtr;

        if(++iter >= numVertices-1) break;
    }
    
    // copy distances back to CPU
    cudaMemcpy(distances, distances_d, sizeof(int)*(numVertices), cudaMemcpyDeviceToHost);

    // copy parent array back to CPU
    cudaMemcpy(parent, parent_d, sizeof(int)*(numVertices), cudaMemcpyDeviceToHost);

    // print time taken
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("SSSP using GPU - Time Taken: %.6f ms \nIterations: %d\n", milliseconds, iter);

    // free up the memory
    free(modified);
    free(finished);
    cudaFree(csrOffsets_d);
    cudaFree(csrCords_d);
    cudaFree(csrWeights_d);
    cudaFree(distances_d);
    cudaFree(parent_d);
    cudaFree(locks_d);
    cudaFree(modified_d);
    cudaFree(modified_next_d);
    cudaFree(finished_d);

    long long sum = 0;
    for(int i=0; i<numVertices; i++) sum += distances[i];
    return sum;
}

// add an edge to the graph and compute SSSP
void addEdge(int u, int v, int w, int &numVertices, int &numEdges, int *distances, int *parent, 
             int *&csrOffsets, int *&csrCords, int *&csrWeights, 
             int *&csrOffsetsR, int *&csrCordsR, int *&csrWeightsR, 
             std::vector<std::vector<int>> &adjList, std::map<std::pair<int,int>, int> &edgeWeights) {
    // find the indices where the edge could be inserted
    int idx1 = find(v, csrCords, csrOffsets[u], csrOffsets[u+1]);
    int idx2 = find(u, csrCordsR, csrOffsetsR[v], csrOffsetsR[v+1]);
    
    // edge already present
    if(idx1 < csrOffsets[u+1] && csrCords[idx1] == v && csrWeights[idx1] != MAX_INT) {
        printf("Warning! Given edge is already present.\n");
        return;
    }

    // edge is getting added
    if(csrWeights[idx1] == MAX_INT) {
        csrWeights[idx1] = w;
        csrWeightsR[idx2] = w;
        edgeWeights[{u, v}] = w;
    } else {    
        numEdges++;
        for(int i=u+1; i<=numVertices; i++) csrOffsets[i]++;
        for(int i=v+1; i<=numVertices; i++) csrOffsetsR[i]++;
        csrCords = (int*)realloc(csrCords, sizeof(int)*numEdges);
        csrWeights = (int*)realloc(csrWeights, sizeof(int)*numEdges);
        csrCordsR = (int*)realloc(csrCordsR, sizeof(int)*numEdges);
        csrWeightsR = (int*)realloc(csrWeightsR, sizeof(int)*numEdges);
        for(int i=numEdges-1; i>=idx1+1; i--) {
            csrCords[i] = csrCords[i-1];
            csrWeights[i] = csrWeights[i-1];
        }
        for(int i=numEdges-1; i>=idx2+1; i--) {
            csrCordsR[i] = csrCordsR[i-1];
            csrWeightsR[i] = csrWeightsR[i-1];
        }
        csrCords[idx1] = v;
        csrWeights[idx1] = w;
        csrCordsR[idx2] = u;
        csrWeightsR[idx2] = w;
        adjList[u].push_back(v);
        edgeWeights[{u,v}] = w;
    }

    // no need to update the distances if the path is not becoming shorter
    if((distances[u]==MAX_INT) || (distances[u]+w >= distances[v])) return;

    // the path is getting reduced so update distance and parent for v
    distances[v] = distances[u] + w;
    parent[v] = u;

    // now propagate the values

    // launch config
    const int numThreads = 1024;
    const int numBlocksV = (numVertices+numThreads-1)/numThreads;
    // const numBlocksE = (numEdges+numOfThreads-1)/numThreads;

    // pointers for arrays on CPU
    bool *modified = (bool*)malloc(sizeof(bool)*numVertices);
    bool *finished = (bool*)malloc(sizeof(bool));
    
    // pointers for arrays on GPU
    int *csrOffsets_d, *csrCords_d, *csrWeights_d;
    int *distances_d, *parent_d, *locks_d;
    bool *modified_d, *modified_next_d, *finished_d;

    // allocate memory on GPU
    cudaMalloc(&csrOffsets_d, sizeof(int)*(numVertices+1));
    cudaMalloc(&csrCords_d, sizeof(int)*(numEdges));
    cudaMalloc(&csrWeights_d, sizeof(int)*(numEdges));
    cudaMalloc(&distances_d, sizeof(int)*numVertices);
    cudaMalloc(&parent_d, sizeof(int)*numVertices);
    cudaMalloc(&locks_d, sizeof(int)*numVertices);
    cudaMalloc(&modified_d, sizeof(bool)*numVertices);
    cudaMalloc(&modified_next_d, sizeof(bool)*numVertices);
    cudaMalloc(&finished_d, sizeof(bool));

    // initialize the CPU arrays
    for(int i=0; i<numVertices; i++) {
        modified[i] = false;
    }
    modified[v] = true;
    *finished = false;

    // for recording the total time taken
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // copy to GPU
    cudaMemcpy(csrOffsets_d, csrOffsets, sizeof(int)*(numVertices+1), cudaMemcpyHostToDevice);
    cudaMemcpy(csrCords_d, csrCords, sizeof(int)*(numEdges), cudaMemcpyHostToDevice);
    cudaMemcpy(csrWeights_d, csrWeights, sizeof(int)*(numEdges), cudaMemcpyHostToDevice);
    cudaMemcpy(distances_d, distances, sizeof(int)*(numVertices), cudaMemcpyHostToDevice);
    cudaMemcpy(parent_d, parent, sizeof(int)*(numVertices), cudaMemcpyHostToDevice);
    cudaMemcpy(modified_d, modified, sizeof(bool)*(numVertices), cudaMemcpyHostToDevice);

    // call kernel to compute edge relaxing till no more updates or at max "numVertices-1" times
    int iter = 0;
    init_kernel<bool><<<numBlocksV, numThreads>>>(modified_next_d, false, numVertices);
    init_kernel<int><<<numBlocksV, numThreads>>>(locks_d, 0, numVertices);
    while(*finished != true) {
        init_kernel<bool><<<1, 1>>>(finished_d, true, 1);
        sssp_kernel<<<numBlocksV, numThreads>>>(csrOffsets_d, csrCords_d, csrWeights_d, distances_d, parent_d, 
                                                locks_d, numVertices, modified_d, modified_next_d, finished_d);
        init_kernel<bool><<<numBlocksV, numThreads>>>(modified_d, false, numVertices);
        cudaDeviceSynchronize();

        // // check for error
        // cudaError_t error = cudaGetLastError();
        // if(error != cudaSuccess) {
        //     printf("CUDA error: %s\n", cudaGetErrorString(error));
        // }

        cudaMemcpy(finished, finished_d, sizeof(bool), cudaMemcpyDeviceToHost);
        bool *tempPtr = modified_next_d;
        modified_next_d = modified_d;
        modified_d = tempPtr;

        if(++iter >= numVertices-1) break;
    }
    
    // copy distances back to CPU
    cudaMemcpy(distances, distances_d, sizeof(int)*(numVertices), cudaMemcpyDeviceToHost);

    // copy parent array back to CPU
    cudaMemcpy(parent, parent_d, sizeof(int)*(numVertices), cudaMemcpyDeviceToHost);

    // print time taken
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("SSSP using GPU - Time Taken: %.6f ms \nIterations: %d\n", milliseconds, iter);

    // free up the memory
    free(modified);
    free(finished);
    cudaFree(csrOffsets_d);
    cudaFree(csrCords_d);
    cudaFree(csrWeights_d);
    cudaFree(distances_d);
    cudaFree(parent_d);
    cudaFree(locks_d);
    cudaFree(modified_d);
    cudaFree(modified_next_d);
    cudaFree(finished_d);
}

// delete an edge from the graph and compute SSSP
void deleteEdge(int u, int v, int &numVertices, int &numEdges, int *distances, int *parent, 
                int *&csrOffsets, int *&csrCords, int *&csrWeights, 
                int *&csrOffsetsR, int *&csrCordsR, int *&csrWeightsR, 
                std::vector<std::vector<int>> &adjList, std::map<std::pair<int,int>, int> &edgeWeights) {
    // find the indices of the edge that needs to be deleted
    int idx1 = find(v, csrCords, csrOffsets[u], csrOffsets[u+1]);
    int idx2 = find(u, csrCordsR, csrOffsetsR[v], csrOffsetsR[v+1]);
    
    // such edge not present
    if(idx1 >= csrOffsets[u+1] || csrCords[idx1] != v) {
        printf("Warning! Given edge is not present.\n");
        return;
    }

    // set the edge weight as infinity (for representing the absence of edge)
    csrWeights[idx1] = MAX_INT;
    csrWeightsR[idx2] = MAX_INT;
    edgeWeights[{u,v}] = MAX_INT;

    // if u is not the parent of v in SPT then return
    if(u!=parent[v]) return;
    
    // now, propagate changes
    // launch config
    const int numThreads = 1024;
    const int numBlocksV = (numVertices+numThreads-1)/numThreads;
    // const numBlocksE = (numEdges+numOfThreads-1)/numThreads;

    // pointers for arrays on CPU
    bool *modified = (bool*)malloc(sizeof(bool)*numVertices);
    bool *finished = (bool*)malloc(sizeof(bool));
    
    // pointers for arrays on GPU
    int *csrOffsetsR_d, *csrCordsR_d, *csrWeightsR_d;
    int *distances_d, *parent_d;
    bool *modified_d, *finished_d;

    // allocate memory on GPU
    cudaMalloc(&csrOffsetsR_d, sizeof(int)*(numVertices+1));
    cudaMalloc(&csrCordsR_d, sizeof(int)*(numEdges));
    cudaMalloc(&csrWeightsR_d, sizeof(int)*(numEdges));
    cudaMalloc(&distances_d, sizeof(int)*numVertices);
    cudaMalloc(&parent_d, sizeof(int)*numVertices);
    cudaMalloc(&modified_d, sizeof(bool)*numVertices);
    cudaMalloc(&finished_d, sizeof(bool));

    // initialize the CPU arrays
    for(int i=0; i<numVertices; i++) modified[i] = false;
    modified[v] = true;
    distances[v] = MAX_INT;
    parent[v] = -1;
    *finished = false;

    // for recording the total time taken
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // copy to GPU
    cudaMemcpy(csrOffsetsR_d, csrOffsetsR, sizeof(int)*(numVertices+1), cudaMemcpyHostToDevice);
    cudaMemcpy(csrCordsR_d, csrCordsR, sizeof(int)*(numEdges), cudaMemcpyHostToDevice);
    cudaMemcpy(csrWeightsR_d, csrWeightsR, sizeof(int)*(numEdges), cudaMemcpyHostToDevice);
    cudaMemcpy(distances_d, distances, sizeof(int)*(numVertices), cudaMemcpyHostToDevice);
    cudaMemcpy(parent_d, parent, sizeof(int)*(numVertices), cudaMemcpyHostToDevice);
    cudaMemcpy(modified_d, modified, sizeof(bool)*(numVertices), cudaMemcpyHostToDevice);

    // set all descendants of v in SPT as unreachable
    int iter1 = 0;
    while(*finished != true) {
        init_kernel<bool><<<1, 1>>>(finished_d, true, 1);
        mark_descendants<<<numBlocksV, numThreads>>>(distances_d, parent_d, modified_d, 
                                                     numVertices, finished_d);

        // // check for error
        // cudaError_t error = cudaGetLastError();
        // if(error != cudaSuccess) {
        //     printf("CUDA error: %s\n", cudaGetErrorString(error));
        // }

        cudaMemcpy(finished, finished_d, sizeof(bool), cudaMemcpyDeviceToHost);
        if(++iter1 >= numVertices-1) break;
    }

    // call kernel to compute edge relaxing till no more updates or at max "numVertices-1" times
    *finished = false;
    int iter2 = 0;
    while(*finished != true) {
        init_kernel<bool><<<1, 1>>>(finished_d, true, 1);
        fetch_and_update<<<numBlocksV, numThreads>>>(csrOffsetsR_d, csrCordsR_d, csrWeightsR_d, distances_d, 
                                                     parent_d, numVertices, modified_d, finished_d);
        cudaDeviceSynchronize();

        // // check for error
        // cudaError_t error = cudaGetLastError();
        // if(error != cudaSuccess) {
        //     printf("CUDA error: %s\n", cudaGetErrorString(error));
        // }

        cudaMemcpy(finished, finished_d, sizeof(bool), cudaMemcpyDeviceToHost);
        if(++iter2 >= numVertices-1) break;
    }

    // set parent as -1 for unreachable nodes
    set_unreachable<<<numBlocksV, numThreads>>>(distances_d, parent_d, numVertices);
    
    // copy distances back to CPU
    cudaMemcpy(distances, distances_d, sizeof(int)*(numVertices), cudaMemcpyDeviceToHost);

    // copy distances back to CPU
    cudaMemcpy(parent, parent_d, sizeof(int)*(numVertices), cudaMemcpyDeviceToHost);

    // print time taken
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time Taken: %.6f ms \nIterations: %d\n", milliseconds, iter1+iter2);

    // free up the memory
    free(modified);
    free(finished);
    cudaFree(csrOffsetsR_d);
    cudaFree(csrCordsR_d);
    cudaFree(csrWeightsR_d);
    cudaFree(distances_d);
    cudaFree(parent_d);
    cudaFree(modified_d);
    cudaFree(finished_d);
}

// binary search - returns the index where arr[index] == key. if key not present 
// in the given range then return first index where arr[index] > key
int find(int key, int *arr, int start, int end) {
    int mid;
    while(start < end) {
        mid = start + (end-start)/2;
        if(arr[mid] == key) return mid;
        else if(arr[mid] < key) start = mid+1;
        else end = mid;
    }
    return start;
}

// kernel invoked in parallel bellman ford sssp routine
__global__ void sssp_kernel(int *csrOffsets_d, int *csrCords_d, int *csrWeights_d, int *distances_d, 
                            int *parent_d, int *locks_d, int numVertices, bool *modified_d, 
                            bool *modified_next_d, bool *finished_d) {
    unsigned int id = blockDim.x*blockIdx.x + threadIdx.x;
    if(id<numVertices && modified_d[id]) {
        int distToCurNode = distances_d[id];
        int v, newDist, lock;
        for(int e=csrOffsets_d[id]; e<csrOffsets_d[id+1] && csrWeights_d[e]!=MAX_INT; e++) {
            v = csrCords_d[e];
            newDist = distToCurNode + csrWeights_d[e];
            do {
                lock = atomicCAS(&locks_d[v], 0, 1);
                if(lock==0 && newDist < distances_d[v]) {
                    distances_d[v] = newDist;
                    parent_d[v] = id;
                    modified_next_d[v] = true;
                    *finished_d = false;
                }
            } while(lock != 0);
            atomicExch(&locks_d[v], 0);
        }
    }
}

// kernel for value initialization
template <typename T>
__global__ void init_kernel(T *array, T val, int arraySize) {
    unsigned int id = blockDim.x*blockIdx.x + threadIdx.x;
    if(id < arraySize) array[id] = val;
}

// kernel for setting all the descendants of v in SPT
__global__ void mark_descendants(int *distances_d, int *parent_d, bool *modified_d, int numVertices,
                                 bool *finished_d) {
    unsigned int id = blockDim.x*blockIdx.x + threadIdx.x;
    if(id<numVertices && modified_d[id]==false) {
        int parent = parent_d[id];
        if(parent>=0 && modified_d[parent]) {
            distances_d[id] = MAX_INT;
            parent_d[id] = -1;
            modified_d[id] = true;
            *finished_d = false;
        }
    }
}

// kernel for updating the distance of modified nodes
__global__ void fetch_and_update(int *csrOffsetsR_d, int *csrCordsR_d, int *csrWeightsR_d, int *distances_d, 
                                 int *parent_d, int numVertices, bool *modified_d, bool *finished_d) {
    unsigned int id = blockDim.x*blockIdx.x + threadIdx.x;
    if(id<numVertices && modified_d[id]) {
        int u;
        for(int e=csrOffsetsR_d[id]; e<csrOffsetsR_d[id+1]; e++) {
            u = csrCordsR_d[e];
            if(distances_d[u] != MAX_INT && csrWeightsR_d[e] != MAX_INT) {
                if(distances_d[id] > distances_d[u]+csrWeightsR_d[e]) {
                    distances_d[id] = distances_d[u]+csrWeightsR_d[e];
                    parent_d[id] = u;
                    *finished_d = false;
                }
            }
        }
    }
}

// kernel for setting parents of unreachable as -1
__global__ void set_unreachable(int *distances_d, int *parent_d, int numVertices) {
    unsigned int id = blockDim.x*blockIdx.x + threadIdx.x;
    if(id<numVertices && distances_d[id]==MAX_INT) {
        parent_d[id] = -1;
    }
}