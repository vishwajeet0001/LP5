#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

#define MAX_NODES 100

vector<vector<int>> adj(MAX_NODES, vector<int>(MAX_NODES));  // Graph representation
vector<bool> visited(MAX_NODES, false);  // Visited nodes

// Parallel BFS function
void parallel_bfs(int start_node, int num_nodes) {
    vector<int> queue;
    queue.push_back(start_node);
    visited[start_node] = true;

    int front = 0;
    while (front < queue.size()) {
        int level_size = queue.size() - front;

        #pragma omp parallel for
        for (int i = 0; i < level_size; i++) {
            int node = queue[front + i];
            cout << "BFS Visited: " << node << endl;

            for (int j = 0; j < num_nodes; j++) {
                if (adj[node][j] && !visited[j]) {
                    visited[j] = true;
                    queue.push_back(j);
                }
            }
        }
        front += level_size;
    }
}

// Parallel DFS function
void parallel_dfs(int node, int num_nodes) {
    visited[node] = true;
    cout << "DFS Visited: " << node << endl;

    #pragma omp parallel for
    for (int i = 0; i < num_nodes; i++) {
        if (adj[node][i] && !visited[i]) {
            parallel_dfs(i, num_nodes);  // Recursive DFS
        }
    }
}

int main() {
    int num_nodes = 6;

    // Example graph (adjacency matrix)
    adj[0][1] = adj[1][0] = 1;
    adj[0][2] = adj[2][0] = 1;
    adj[1][3] = adj[3][1] = 1;
    adj[1][4] = adj[4][1] = 1;
    adj[2][5] = adj[5][2] = 1;

    // Parallel BFS
    cout << "Parallel BFS:\n";
    parallel_bfs(0, num_nodes);

    // Reset visited array
    fill(visited.begin(), visited.end(), false);

    // Parallel DFS
    cout << "\nParallel DFS:\n";
    parallel_dfs(0, num_nodes);

    return 0;
}
