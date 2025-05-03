#include <iostream>
#include <vector>
#include <stack>
#include <queue>
#include <chrono>
#include <random>
#include <omp.h>

using namespace std;

class Graph
{
public:
    int vertices;
    vector<vector<int>> graph;
    vector<bool> visited;

    Graph(int v, int e) : vertices(v)
    {
        graph.resize(vertices);
        generate_random_graph(e);
    }

    void generate_random_graph(int edges)
    {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<int> dist(0, vertices - 1);

        for (int i = 0; i < edges; i++)
        {
            int u = dist(gen);
            int v = dist(gen);
            if (u != v)
            {
                graph[u].push_back(v);
                graph[v].push_back(u);
            }
        }
    }

    void initialize_visited()
    {
        visited.assign(vertices, false);
    }

    void parallel_dfs(int start)
    {
        visited[start] = true;
#pragma omp parallel
        {
            stack<int> local_stack;
            local_stack.push(start);

            while (!local_stack.empty())
            {
                int current;
#pragma omp critical
                {
                    if (!local_stack.empty())
                    {
                        current = local_stack.top();
                        local_stack.pop();
                    }
                    else
                    {
                        current = -1;
                    }
                }

                if (current == -1)
                    continue;

                for (int neighbor : graph[current])
                {
                    if (!visited[neighbor])
                    {
                        visited[neighbor] = true;
#pragma omp critical
                        {
                            local_stack.push(neighbor);
                        }
                    }
                }
            }
        }
    }

    
    void parallel_bfs(int start)
    {
        visited[start] = true;
        queue<int> q;
        q.push(start);

#pragma omp parallel
        {
            queue<int> local_queue;
            while (!q.empty())
            {
                int current;
#pragma omp critical
                {
                    if (!q.empty())
                    {
                        current = q.front();
                        q.pop();
                    }
                    else
                    {
                        current = -1;
                    }
                }

                if (current == -1)
                    continue;

                for (int neighbor : graph[current])
                {
                    if (!visited[neighbor])
                    {
                        visited[neighbor] = true;
#pragma omp critical
                        {
                            local_queue.push(neighbor);
                        }
                    }
                }

#pragma omp critical
                {
                    while (!local_queue.empty())
                    {
                        q.push(local_queue.front());
                        local_queue.pop();
                    }
                }
            }
        }
    }

    void dfs(int start)
    {
        stack<int> s;
        s.push(start);
        visited[start] = true;
        while (!s.empty())
        {
            int current = s.top();
            s.pop();
            for (int neighbor : graph[current])
            {
                if (!visited[neighbor])
                {
                    s.push(neighbor);
                    visited[neighbor] = true;
                }
            }
        }
    }

    void bfs(int start)
    {
        queue<int> q;
        q.push(start);
        visited[start] = true;
        while (!q.empty())
        {
            int current = q.front();
            q.pop();
            for (int neighbor : graph[current])
            {
                if (!visited[neighbor])
                {
                    q.push(neighbor);
                    visited[neighbor] = true;
                }
            }
        }
    }
};

int main()
{
    int V = 1000000, E = 10000000;
    Graph g(V, E);
    double time_seq_dfs, time_par_dfs, time_seq_bfs, time_par_bfs;

    g.initialize_visited();
    auto start = chrono::high_resolution_clock::now();
    g.dfs(0);
    auto end = chrono::high_resolution_clock::now();
    time_seq_dfs = chrono::duration<double>(end - start).count();
    cout << "Time required for DFS sequential is:" << time_seq_dfs << endl;
    g.initialize_visited();
    start = chrono::high_resolution_clock::now();
    g.parallel_dfs(0);
    end = chrono::high_resolution_clock::now();
    time_par_dfs = chrono::duration<double>(end - start).count();
    cout << "Time required for DFS parallel is:" << time_par_dfs << endl;
    g.initialize_visited();
    start = chrono::high_resolution_clock::now();
    g.bfs(0);
    end = chrono::high_resolution_clock::now();
    time_seq_bfs = chrono::duration<double>(end - start).count();
    cout << "Time required for BFS sequential is:" << time_seq_bfs << endl;
    g.initialize_visited();
    start = chrono::high_resolution_clock::now();
    g.parallel_bfs(0);
    end = chrono::high_resolution_clock::now();
    time_par_bfs = chrono::duration<double>(end - start).count();
    cout << "Time required for BFS parallel is:" << time_par_bfs << endl;
    cout << "DFS Speedup: " << (time_par_dfs > 0 ? time_seq_dfs / time_par_dfs : 0) << endl;
    cout << "BFS Speedup: " << (time_par_bfs > 0 ? time_seq_bfs / time_par_bfs : 0) << endl;

    return 0;
}
