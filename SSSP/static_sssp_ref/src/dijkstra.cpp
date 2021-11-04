#include "../include/graph.h"

// host function to compute shortest path using dijkstra's algorithm
long long Dijkstra(int numVertices, const std::vector<std::vector<int>> &adjList, 
             const std::map<std::pair<int,int>, int> &weights, 
             int *distances, int source=0) {
    for(int i=0; i<numVertices; i++) distances[i] = INT_MAX>>1;
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
