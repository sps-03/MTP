#include <vector>
#include <set>
#include <utility>
#include <iostream>
#include <chrono>
#include <random>

// main function
int main(int argc, char **argv) {
    // if input, or update file names are not provided then exit
    if(argc != 3) {
        printf("Enter the input and update file path in the command line.\n");
        return 0;
    }

    // read file names
    char *inputFile = argv[1];
    char *updateFile = argv[2];    

    // open input file
    FILE *inputFilePtr = fopen(inputFile, "r");
    FILE *updateFilePtr = fopen(updateFile, "w");
    
    // if not able to open the input file then exit
    if(inputFilePtr == NULL) {
        printf("Failed to open the input file.\n");
        return 0;
    }

    // read num vertices and num edges
    int numVertices, numEdges, startVertex;
    fscanf(inputFilePtr, "%d", &numVertices);
    fscanf(inputFilePtr, "%d", &numEdges);
    fscanf(inputFilePtr, "%d", &startVertex);

    // get max update operations required (30% of initial numEdges)
    int maxUpdates = (int) (0.30*numEdges);

    // percentage of edge addition operations
    int percentAdd = 50;

    // to store all the edges present in the graph
    std::vector<std::pair<int, int>> edges_v;

    // to store all the edges present in the graph
    std::set<std::pair<int, int>> edges_s;

    // read edges from the input file
    for(int i=0; i<numEdges; i++) {
        int src, dest;
        fscanf(inputFilePtr, "%d %d", &src, &dest);
        edges_v.push_back({src, dest});
        edges_s.insert({src, dest});
    }

    // close the input file
    fclose(inputFilePtr);

    // get the seed - time between the system clock (present time) and clock's epoch
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
     
    // mt19937 is a standard mersenne_twister_engine
    std::mt19937 generator (seed);
    
    // to keep track of edes which got updated (to eliminate reduntant updates)
    std::set<std::pair<int,int>> updt_edges;

    // to keep track of actual number of additions and deletions
    int numAdds=0, numDels=0;
    
    // start populating the updates
    srand(3);
    for(int i=0; i<maxUpdates; i++) {
        int u, v;
        if(rand()%100 < percentAdd) {
            do {
                u = generator() % numVertices;
                v = generator() % numVertices;
            } while(updt_edges.find({u, v})!=updt_edges.end() || edges_s.find({u, v})!=edges_s.end());
            fprintf(updateFilePtr, "a %d %d\n", u, v);
            numAdds++;
        } else {
            do {
                int idx = generator() % numEdges;
                u = edges_v[idx].first;
                v = edges_v[idx].second;
            } while(updt_edges.find({u, v})!=updt_edges.end());
            fprintf(updateFilePtr, "d %d %d\n", u, v);
            numDels++;
        }            
        updt_edges.insert({u, v});
    }

    // print the number of actual additions and deletes
    printf("Additions=%d, Deletions=%d\n", numAdds, numDels);

    // close update file
    fclose(updateFilePtr);

    return 0;
}