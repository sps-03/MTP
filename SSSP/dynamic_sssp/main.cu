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
                   int *distances, int source);
long long dijkstra(int numVertices, const std::vector<std::vector<int>> &adjList, 
                   const std::map<std::pair<int,int>, int> &weights, int *distances, int source);
bool checkDistances(int *distances_g, int *distances_c, int numVertices);
void printArray(int *arr, int len);
int find(int dest, int *csrCords, int start, int end);
void addEdge(int u, int v, int w, int &numVertices, int &numEdges, int *&distances,
             int *&csrOffsets, int *&csrCords, int *&csrWeights, 
             int *&csrOffsetsR, int *&csrCordsR, int *&csrWeightsR, 
             std::vector<std::vector<int>> &adjList, std::map<std::pair<int,int>, int> &weights, int source);
void deleteEdge(int u, int v, int &numVertices, int &numEdges, int *&distances,
                int *&csrOffsets, int *&csrCords, int *&csrWeights, 
                int *&csrOffsetsR, int *&csrCordsR, int *&csrWeightsR, 
                std::vector<std::vector<int>> &adjList, std::map<std::pair<int,int>, int> &weights, int source);
void update(int startVertex, int numVertices, int numEdges, int *csrOffsets, int *csrCords, int *csrWeights, 
            int *csrOffsetsR, int *csrCordsR, int *csrWeightsR, int *distances);


// DEVICE FUNCTIONS - DECLARATION:
template <typename T>
__global__ void init_kernel(T *array, T val, int arraySize);
__global__ void sssp_kernel(int *csrOffsets_g, int *csrCords_g, int *csrWeights_g, int *distances_g, int numVertices, bool *modified_g, bool *modified_next_g, bool *finished_g);
__global__ void fetch_and_update(int *csrOffsetsR_g, int *csrCordsR_g, int *csrWeightsR_g, int *distances_g, int *distances_next_g, int numVertices, bool *modified_g, bool *modified_next_g, bool *finished_g);
__global__ void set_array_vals(int *csrOffsets_g, int *csrCords_g, int numVertices, bool *modified_g, bool *modified_next_g, int *distances_next_g);
__global__ void copy_D2D(int *distances_g, int *distances_next_g, int numVertices);

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
    std::map<std::pair<int,int>, int> weights;

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
        weights[{src,dest}] = weight;

        // // for undirected graphs
        // COO[numEdges+i].src = dest;
        // COO[numEdges+i].dest = src;
        // COO[numEdges+i].weight = weight;
        // adjList[dest].push_back(src);
        // weights[{dest,src}] = weight;
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
    for(int i=0; i<numVertices+1; i++) csrOffsets[i] = 0;

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
    for(int i=0; i<numVertices+1; i++) csrOffsetsR[i] = 0;

    // update the Coordinates and Weights array
    for(int i=0; i<numEdges; i++) {
        csrCordsR[i] = COO[i].src;
        csrWeightsR[i] = COO[i].weight;
    }

    // update the Offsets array
    for(int i=0; i<numEdges; i++) csrOffsetsR[COO[i].dest+1]++;		//store the frequency
    for(int i=0; i<numVertices; i++) csrOffsetsR[i+1] += csrOffsetsR[i];	// do cumulative sum
    
    // converting the graph to CSRs done

    // shortest distances from start vertex
    // distances_g is computed using GPU and distances_c is computed using CPU
    int *distances_g = (int*)malloc(sizeof(int)*numVertices);
    int *distances_c = (int*)malloc(sizeof(int)*numVertices);

    // compute the shortest paths
    long long gpuTotalPathSum = SSSP_GPU(numVertices, numEdges, csrOffsets, csrCords, csrWeights, distances_g, startVertex);
    long long cpuTotalPathSum = dijkstra(numVertices, adjList, weights, distances_c, startVertex);

    // check for path sum
    if(gpuTotalPathSum != cpuTotalPathSum) {
        printf("Initial Graph: Difference in CPU & GPU paths.!!!\n");
        return 0;
    }

    // check whether both the distances arrays are same or not
    if(checkDistances(distances_g, distances_c, numVertices) == false) {
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
        if(distances_g[i]==MAX_INT)
            fprintf(outputFilePtr, "The distance to vertex %d is INF\n", i);
        else
            fprintf(outputFilePtr, "The distance to vertex %d is %d\n", i, distances_g[i]);
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

            addEdge(u, v, w, numVertices, numEdges, distances_g,
                    csrOffsets, csrCords, csrWeights, csrOffsetsR, csrCordsR, csrWeightsR, 
                    adjList, weights, startVertex);
        } else if(type=='d') {
            fscanf(updateFilePtr, "%d", &u);
            fscanf(updateFilePtr, "%d", &v);
            fprintf(outputFilePtr, "%c %d %d\n", type, u, v);
            printf("%c %d %d\n", type, u, v);

            deleteEdge(u, v, numVertices, numEdges, distances_g,
                       csrOffsets, csrCords, csrWeights, csrOffsetsR, csrCordsR, csrWeightsR, 
                       adjList, weights, startVertex);
        }

        // write the result to output file
        for(int i=0; i<numVertices; i++) {
            if(distances_g[i]==MAX_INT)
                fprintf(outputFilePtr, "The distance to vertex %d is INF\n", i);
            else
                fprintf(outputFilePtr, "The distance to vertex %d is %d\n", i, distances_g[i]);
        }
        fprintf(outputFilePtr, "\n");
    }
    
    // final checking
    distances_c = (int*)realloc(distances_c, sizeof(int)*numVertices);
    cpuTotalPathSum = dijkstra(numVertices, adjList, weights, distances_c, startVertex);
    gpuTotalPathSum = 0; for(int i=0; i<numVertices; i++) gpuTotalPathSum += distances_g[i];

    // check for total path sum
    if(gpuTotalPathSum != cpuTotalPathSum) {
        printf("\nFinal: Difference in CPU & GPU paths.!!!\n");
        return 0;
    }

    // check whether both the distances arrays are same or not
    if(checkDistances(distances_g, distances_c, numVertices) == false) {
        printf("\nFinal: Check failed..!!!\n");
        return 0;
    }

    // print success message
    printf("\nComputed SSSP for final graph successfully.\n");

    // write the final result to output file
    fprintf(outputFilePtr, "Distances for final graph\n");
    for(int i=0; i<numVertices; i++) {
        if(distances_g[i]==MAX_INT)
            fprintf(outputFilePtr, "The distance to vertex %d is INF\n", i);
        else
            fprintf(outputFilePtr, "The distance to vertex %d is %d\n", i, distances_g[i]);
    }
    fprintf(outputFilePtr, "\n");

    // free memory allocated on CPU
    free(csrOffsets);
    free(csrCords);
    free(csrWeights);
    free(csrOffsetsR);
    free(csrCordsR);
    free(csrWeightsR);
    free(distances_g);
    free(distances_c);

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
bool checkDistances(int *distances_g, int *distances_c, int numVertices) {
    for(int i=0; i<numVertices; i++) {
        if(distances_g[i] != distances_c[i]) return false;
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
                   const std::map<std::pair<int,int>, int> &weights, int *distances, int source=0) {
    for(int i=0; i<numVertices; i++) distances[i] = MAX_INT;
    distances[source] = 0;

    std::set<std::pair<int,int>> active_vertices;
    active_vertices.insert({0, source});

    while(!active_vertices.empty()) {
        int u = active_vertices.begin()->second;
        active_vertices.erase(active_vertices.begin());
        for(int v : adjList[u]) {
            if(weights.at({u,v}) == MAX_INT) continue;
            int newDist = distances[u] + weights.at({u,v});
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

// host function for parallel bellman ford (modified) routine
long long SSSP_GPU(int numVertices, int numEdges, int *csrOffsets, int *csrCords, int *csrWeights, 
                   int *distances, int source=0) {
    // launch config
    const int numThreads = 1024;
    const int numBlocksV = (numVertices+numThreads-1)/numThreads;
    // const numBlocksE = (numEdges+numOfThreads-1)/numThreads;

    // pointers for arrays on CPU
    bool *modified = (bool*)malloc(sizeof(bool)*numVertices);
    bool *finished = (bool*)malloc(sizeof(bool));
    
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

    // initialize the CPU arrays
    for(int i=0; i<numVertices; i++) {
        distances[i] = MAX_INT;
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
        //     printf("CUDA error: %s\n", cudaGetErrorString(error));
        // }

        cudaMemcpy(finished, finished_g, sizeof(bool), cudaMemcpyDeviceToHost);
        bool *tempPtr = modified_next_g;
        modified_next_g = modified_g;
        modified_g = tempPtr;

        if(++iter >= numVertices-1) break;
    }
    
    // copy distances back to CPU
    cudaMemcpy(distances, distances_g, sizeof(int)*(numVertices), cudaMemcpyDeviceToHost);

    // print time taken
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("SSSP using GPU - Time Taken: %.6f ms \nIterations: %d\n", milliseconds, iter);

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

// add an edge to the graph and compute SSSP
void addEdge(int u, int v, int w, int &numVertices, int &numEdges, int *&distances,
             int *&csrOffsets, int *&csrCords, int *&csrWeights, 
             int *&csrOffsetsR, int *&csrCordsR, int *&csrWeightsR, 
             std::vector<std::vector<int>> &adjList, std::map<std::pair<int,int>, int> &weights, int source) {
    // find the indices where the edge could be inserted
    int idx1 = find(v, csrCords, csrOffsets[u], csrOffsets[u+1]);
    int idx2 = find(u, csrCordsR, csrOffsetsR[v], csrOffsetsR[v+1]);
    
    // edge already present
    if(idx1 < csrOffsets[u+1] && csrCords[idx1] == v) {
        // exact same edge is present
        if(csrWeights[idx1] == w) {
            printf("Given edge with same weight already present\n");
            return;
        }

        // edge present but weight is getting updated
        csrWeights[idx1] = w;
        csrWeightsR[idx2] = w;
        weights[{u,v}] = w;

        if(v==source) return;
        
        update(v, numVertices, numEdges, csrOffsets, csrCords, csrWeights, 
               csrOffsetsR, csrCordsR, csrWeightsR, distances);
    }
    // new edge is getting added
    else {
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
        weights[{u,v}] = w;

        if(v==source) return;
        
        update(v, numVertices, numEdges, csrOffsets, csrCords, csrWeights, 
               csrOffsetsR, csrCordsR, csrWeightsR, distances);
    }
}

// delete an edge from the graph and compute SSSP
void deleteEdge(int u, int v, int &numVertices, int &numEdges, int *&distances,
                int *&csrOffsets, int *&csrCords, int *&csrWeights, 
                int *&csrOffsetsR, int *&csrCordsR, int *&csrWeightsR, 
                std::vector<std::vector<int>> &adjList, std::map<std::pair<int,int>, int> &weights, int source) {
    // find the indices of the edge that needs to be deleted
    int idx1 = find(v, csrCords, csrOffsets[u], csrOffsets[u+1]);
    int idx2 = find(u, csrCordsR, csrOffsetsR[v], csrOffsetsR[v+1]);
    
    // such edge not present
    if(idx1 >= csrOffsets[u+1] || csrCords[idx1] != v) return;

    csrWeights[idx1] = MAX_INT;
    csrWeightsR[idx2] = MAX_INT;
    weights[{u,v}] = MAX_INT;

    if(v==source) return;
    
    update(v, numVertices, numEdges, csrOffsets, csrCords, csrWeights, 
           csrOffsetsR, csrCordsR, csrWeightsR, distances);
}

void update(int startVertex, int numVertices, int numEdges, 
            int *csrOffsets, int *csrCords, int *csrWeights, 
            int *csrOffsetsR, int *csrCordsR, int *csrWeightsR, int *distances) {
    // launch config
    const int numThreads = 1024;
    const int numBlocksV = (numVertices+numThreads-1)/numThreads;
    // const numBlocksE = (numEdges+numOfThreads-1)/numThreads;

    // pointers for arrays on CPU
    bool *modified = (bool*)malloc(sizeof(bool)*numVertices);
    bool *finished = (bool*)malloc(sizeof(bool));
    
    // pointers for arrays on GPU
    int *csrOffsets_g, *csrCords_g, *csrWeights_g;
    int *csrOffsetsR_g, *csrCordsR_g, *csrWeightsR_g;
    int *distances_g, *distances_next_g;
    bool *modified_g, *modified_next_g, *finished_g;

    // allocate memory on GPU
    cudaMalloc(&csrOffsets_g, sizeof(int)*(numVertices+1));
    cudaMalloc(&csrCords_g, sizeof(int)*(numEdges));
    cudaMalloc(&csrWeights_g, sizeof(int)*(numEdges));
    cudaMalloc(&csrOffsetsR_g, sizeof(int)*(numVertices+1));
    cudaMalloc(&csrCordsR_g, sizeof(int)*(numEdges));
    cudaMalloc(&csrWeightsR_g, sizeof(int)*(numEdges));
    cudaMalloc(&distances_g, sizeof(int)*numVertices);
    cudaMalloc(&distances_next_g, sizeof(int)*numVertices);
    cudaMalloc(&modified_g, sizeof(bool)*numVertices);
    cudaMalloc(&modified_next_g, sizeof(bool)*numVertices);
    cudaMalloc(&finished_g, sizeof(bool));

    // initialize the CPU arrays
    for(int i=0; i<numVertices; i++) modified[i] = false;
    modified[startVertex] = true;
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
    cudaMemcpy(csrOffsetsR_g, csrOffsetsR, sizeof(int)*(numVertices+1), cudaMemcpyHostToDevice);
    cudaMemcpy(csrCordsR_g, csrCordsR, sizeof(int)*(numEdges), cudaMemcpyHostToDevice);
    cudaMemcpy(csrWeightsR_g, csrWeightsR, sizeof(int)*(numEdges), cudaMemcpyHostToDevice);
    cudaMemcpy(distances_g, distances, sizeof(int)*(numVertices), cudaMemcpyHostToDevice);
    distances[startVertex] = MAX_INT;
    cudaMemcpy(distances_next_g, distances, sizeof(int)*(numVertices), cudaMemcpyHostToDevice);
    cudaMemcpy(modified_g, modified, sizeof(bool)*(numVertices), cudaMemcpyHostToDevice);
    cudaMemcpy(finished_g, finished, sizeof(bool), cudaMemcpyHostToDevice);

    // call kernel to compute edge relaxing till no more updates or at max "numVertices-1" times
    int iter = 0;
    init_kernel<bool><<<numBlocksV, numThreads>>>(modified_next_g, false, numVertices);
    while(*finished != true) {
        *finished = true;
        init_kernel<bool><<<1, 1>>>(finished_g, true, 1);
        fetch_and_update<<<numBlocksV, numThreads>>>(csrOffsetsR_g, csrCordsR_g, csrWeightsR_g, distances_g, distances_next_g, numVertices, modified_g, modified_next_g, finished_g);
        copy_D2D<<<numBlocksV, numThreads>>>(distances_g, distances_next_g, numVertices);
        set_array_vals<<<numBlocksV, numThreads>>>(csrOffsets_g, csrCords_g, numVertices, modified_g, modified_next_g, distances_next_g);
        cudaMemcpy(finished, finished_g, sizeof(bool), cudaMemcpyDeviceToHost);

        // // check for error
        // cudaError_t error = cudaGetLastError();
        // if(error != cudaSuccess) {
        //     printf("CUDA error: %s\n", cudaGetErrorString(error));
        // }

        if(++iter >= numVertices-1) break;
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
    cudaFree(csrOffsetsR_g);
    cudaFree(csrCordsR_g);
    cudaFree(csrWeightsR_g);
    cudaFree(distances_g);
    cudaFree(modified_g);
    cudaFree(modified_next_g);
    cudaFree(finished_g);
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
__global__ void sssp_kernel(int *csrOffsets_g, int *csrCords_g, int *csrWeights_g, int *distances_g, 
                            int numVertices, bool *modified_g, bool *modified_next_g, bool *finished_g) {
    unsigned int id = blockDim.x*blockIdx.x + threadIdx.x;
    if(id<numVertices && modified_g[id]) {
        int distToID = distances_g[id];
        for(int e=csrOffsets_g[id]; e<csrOffsets_g[id+1]; e++) {
            int v = csrCords_g[e];
            int newDist = distToID + csrWeights_g[e];
            if(newDist < distances_g[v]) {
                atomicMin(&distances_g[v], newDist);
                modified_next_g[v] = true;
                *finished_g = false;
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

// kernel for updating the distance of modified nodes
__global__ void fetch_and_update(int *csrOffsetsR_g, int *csrCordsR_g, int *csrWeightsR_g, 
                                 int *distances_g, int *distances_next_g, int numVertices, 
                                 bool *modified_g, bool *modified_next_g, bool *finished_g) {
    unsigned int id = blockDim.x*blockIdx.x + threadIdx.x;
    if(id<numVertices && modified_g[id]) {
        for(int e=csrOffsetsR_g[id]; e<csrOffsetsR_g[id+1]; e++) {
            int u = csrCordsR_g[e];
            if(distances_next_g[u] != MAX_INT && csrWeightsR_g[e] != MAX_INT) 
                atomicMin(&distances_next_g[id], distances_next_g[u]+csrWeightsR_g[e]);
        }
        if(distances_g[id] != distances_next_g[id]) {
            modified_next_g[id] = true;
            *finished_g = false;
        }
        modified_g[id] = false;
    }
}

// kernel for setting modified flags
__global__ void set_array_vals(int *csrOffsets_g, int *csrCords_g, int numVertices, bool *modified_g, 
                               bool *modified_next_g, int *distances_next_g) {
    unsigned int id = blockDim.x*blockIdx.x + threadIdx.x;
    if(id<numVertices && modified_next_g[id]) {
        modified_next_g[id] = false;
        for(int e=csrOffsets_g[id]; e<csrOffsets_g[id+1]; e++) {
            int v = csrCords_g[e];
            modified_g[v] = true;
            distances_next_g[v] = MAX_INT;
        }
    }
}

// kernel for copying arrays values from device to device
__global__ void copy_D2D(int *distances_g, int *distances_next_g, int numVertices) {
    unsigned int id = blockDim.x*blockIdx.x + threadIdx.x;
    if(id<numVertices) {
        distances_g[id] = distances_next_g[id];
    }
}