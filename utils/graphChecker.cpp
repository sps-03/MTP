/*
usage: ./graphChecker inputFileLoc outputFileLoc graphType

inputFileLoc    - path to input file
outputFileLoc   - path to output file
graphType       - can take value "directed" or "undirected"
*/

#include <bits/stdc++.h>
using namespace std;

int main(int argc, char **argv) {
    if(argc != 4) {
        cerr << "ARG ERROR: provide all the required arguments\n";
        cerr << "Usage: ./graphChecker inputFileLoc outputFileLoc graphType\n";
        return 1;
    }

    ios_base::sync_with_stdio(false);
    cin.tie(NULL); cout.tie(NULL);

    char *inputFile = argv[1];
    char *outputFile = argv[2];
    char *graphType = argv[3];

    if(freopen(inputFile, "r", stdin) == NULL) {
        cerr << "ARG ERROR: enter valid input file path\n";
        return 1;
    }
    freopen(outputFile, "w", stdout);
    if(strcmp(graphType, "directed")!=0 && strcmp(graphType, "undirected")!=0) {
        cerr << "ARG ERROR: graphType should be either \"directed\" or \"undirected\"\n";
        return 1;
    }
    
    long long numVertices, numEdges;
    cin >> numVertices >> numEdges;

    map<pair<long long, long long>, long long> mp;
    long long u, v;
    long long countEdges=0, minV=LONG_LONG_MAX, maxV=LONG_LONG_MIN;
    bool selfLoop=false, multi=false;
    long long forward=0, backward=0;
    
    while(cin >> u >> v) {
        countEdges++;
        minV = min(minV, min(u,v));
        maxV = max(maxV, max(u,v));

        int c1 = mp[{u,v}]++;
        if(u == v) selfLoop = true;
        if(c1) multi = true;
        if(c1 == mp[{v,u}]) forward++;
        else backward++;
    }

    cout << "edgeCount=" << countEdges << "\n";
    cout << "minV=" << minV << " maxV=" << maxV << "\n";
    if(selfLoop) cout << "self loop present\n";
    if(multi) cout << "multi-edge present\n";
    if(strcmp(graphType, "undirected")==0 && forward != backward)
        cout << "improper undirected graph\n";

    return 0;
}