# Counting Triangeles in Random Graphs

In a random graph $G(n, p)$ where $p = d/n$ ( $d$ is a constant). We want to count the number of triangles.

This can be sped up by more than 80x by means of *systems* and *mathematics/algorithms*.

My feeling is that, sometimes, in order to obtain high performance, **abstractions** and **modularity** should be sacrificed to allow larger optimization space, which is contradictory to the principles of *software engineering*.

## Requirements
1. GCC with C++17
2. AVX (useless in the fastest version)
3. OpenMP
