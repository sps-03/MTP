#include"../include/graph.h"

// comparator function
bool compareTwoEdges(edgeInfo a, edgeInfo b) {
	if(a.src != b.src) return a.src < b.src;
	return a.dest < b.dest;
}

// host function to check whether two distance arrays are same of not
bool check(int *distance_1, int *distance_2, int numVertices) {
	for(int i=0; i<numVertices; i++) {
        if(distance_1[i] != distance_2[i]) {
			return false;
		}
    }
	return true;
}

// host function to print the distance of each vertex
void printDistances(int *distances, int numVertices) {
    for(int i=0; i<numVertices; i++) {
        printf("The distance of vertex %d is %d\n", i, distances[i]);
    }
}