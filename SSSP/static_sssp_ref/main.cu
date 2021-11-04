#include <bits/stdc++.h>
#include "include/graph.cuh"

// main function
int main(int argc, char **argv) {
    // if input or output file name is not provided then exit
    if(argc != 3) {
        printf("Enter the input and output filepath in the command line.\n");
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

    // Data Structures for storing the graph (as adjacency list) and edge weights
    std::vector<std::vector<int>> adjList(numVertices, std::vector<int>());
    std::map<std::pair<int,int>, int> weights;

    // store the input graph in COO format
	std::vector<edgeInfo> COO(numEdges);
	
	// read from the input file and populate the COO
	for(int i=0; i<numEdges; i++) {
        int src, dest, weight;
        fscanf(inputFilePtr, "%d %d %d", &src, &dest, &weight);

        COO[i].src = src;
        COO[i].dest = dest;
        COO[i].weight = weight;

        adjList[src].push_back(dest);
        weights[{src,dest}] = weight;
    }

    // sort the COO 
	std::sort(COO.begin(), COO.end(), compareTwoEdges);

    // converting the graph in COO format to CSR format
    int* csrOffsets = (int*)malloc(sizeof(int)*(numVertices+1));
    int* csrCords = (int*)malloc(sizeof(int)*(numEdges));
    int* csrWeights = (int*)malloc(sizeof(int)*(numEdges));

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
    int* distance_1 = (int*)malloc(sizeof(int)*numVertices);
    int* distance_2 = (int*)malloc(sizeof(int)*numVertices);

    // compute the shortest paths
    long long gpuTotalPathSum = SSSP_GPU(numVertices, numEdges, csrOffsets, csrCords, csrWeights, distance_1, startVertex);
    long long cpuTotalPathSum = Dijkstra(numVertices, adjList, weights, distance_2, startVertex);
    
    /////////////////////////////////////////////////////////////////////////////
    printf("\n----------------------------------------\n");
    printf("SSSP_GPU: ");
    for(int i=0; i<numVertices; i++) printf("%d ", distance_1[i]);
    printf("\n");
    printf("SSSP_CPU: ");
    for(int i=0; i<numVertices; i++) printf("%d ", distance_2[i]);
    printf("\n");
    printf("----------------------------------------\n");
    /////////////////////////////////////////////////////////////////////////////

    if(gpuTotalPathSum != cpuTotalPathSum) {
        printf("Difference in CPU & GPU paths.!!!\n");
        return 0;
    }

    // check whether both the distance arrays are same or not
    if(check(distance_1, distance_2, numVertices) == false) {
        printf("Check failed..!!!\n");
        return 0;
    }

    // print success message
    printf("Computed SSSP successfully. Check outputs/ directory for results.\n");

    // print the shortest path distances
    // printDistances(distance_1, numVertices);

    // write the result to output file
    for(int i=0; i<numVertices; i++) {
	if(distance_1[i]==1073741823) {
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
