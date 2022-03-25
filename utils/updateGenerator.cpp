/*
usage: ./updateGenerator inputFileLoc updateFileLoc graphType

inputFileLoc    - path to input file
updateFileLoc   - path to update file
graphType       - can take value "directed" or "undirected"
*/

#include <vector>
#include <set>
#include <utility>
#include <iostream>
#include <chrono>
#include <random>
#include <string.h>
using namespace std;

int main(int argc, char **argv) {
    if(argc != 4) {
        cerr << "ARG ERROR: provide all the required arguments\n";
        cerr << "Usage: ./updateGenerator inputFileLoc updateFileLoc graphType\n";
        return 1;
    }

    // ios_base::sync_with_stdio(false);
    // cin.tie(NULL); cout.tie(NULL);

    char *inputFile = argv[1];
    char *updateFile = argv[2];
    char *graphType = argv[3];

    if(freopen(inputFile, "r", stdin) == NULL) {
        cerr << "ARG ERROR: enter valid input file path\n";
        return 1;
    }
    freopen(updateFile, "w", stdout);
    if(strcmp(graphType, "directed")!=0 && strcmp(graphType, "undirected")!=0) {
        cerr << "ARG ERROR: graphType should be either \"directed\" or \"undirected\"\n";
        return 1;
    }

    long long numVertices, numEdges;
    cin >> numVertices >> numEdges;

    // get max update operations required (30% of initial numEdges)
    long long maxUpdates = (long long) (0.30*numEdges);

    // percentage of edge addition operations
    long long percentAdd = 50;

    // to store all the edges present in the graph
    vector<pair<long long, long long>> edges_v;

    // to store all the edges present in the graph
    set<pair<long long, long long>> edges_s;

    // read edges from the input file
    for(long long i=0; i<numEdges; i++) {
        long long src, dest;
        cin >> src >> dest;
        edges_v.push_back({src, dest});
        edges_s.insert({src, dest});
    }

    // get the seed - time between the system clock (present time) and clock's epoch
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
     
    // mt19937 is a standard mersenne_twister_engine
    mt19937 generator (seed);
    
    // to keep track of edes which got updated (to eliminate reduntant updates)
    set<pair<long long,long long>> edges_updated;

    // to keep track of actual number of additions and deletions
    long long numAdds=0, numDels=0;
    
    // start populating the updates
    srand(3);
    for(long long i=0; i<maxUpdates; i++) {
        long long u, v;
        if(rand()%100 < percentAdd) {
            do {
                u = generator() % numVertices;
                v = generator() % numVertices;
            } while(edges_updated.find({u, v})!=edges_updated.end() || edges_s.find({u, v})!=edges_s.end());
            
            cout << "a " << u << " " << v << "\n";
            numAdds++;
        }
        else {
            do {
                long long idx = generator() % numEdges;
                u = edges_v[idx].first;
                v = edges_v[idx].second;
            } while(edges_updated.find({u, v})!=edges_updated.end());

            cout << "d " << u << " " << v << "\n";
            numDels++;
        }
        edges_updated.insert({u, v});
        if(strcmp(graphType, "undirected")==0) edges_updated.insert({v, u});
    }

    clog << "Additions=" << numAdds << ", Deletions=" << numDels << "\n";
    return 0;
}