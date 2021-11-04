#ifndef __GRAPH__HEADER__
#define __GRAPH__HEADER__
#include <bits/stdc++.h>
#include <stdio.h>
#include <utility>

// structure for storing edge information
struct edgeInfo {
	int src, dest;
    int weight;
};

// comparator function
bool compareTwoEdges(edgeInfo a, edgeInfo b);

// function prototypes for host functions
long long SSSP_GPU(int numVertices, int numEdges, int *csrOffsets, int *csrCords, 
                   int *csrWeights, int *distances, int source);
long long Dijkstra(int numVertices, const std::vector<std::vector<int>> &adjList, 
                   const std::map<std::pair<int,int>, int> &weights, 
                   int *distances, int source);
bool check(int *distance_1, int *distance_2, int numVertices);
void printDistances(int *distances, int numVertices);

#endif