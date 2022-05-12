#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>

inline static std::unique_ptr<std::unique_ptr<bool[]>[]> GenRandomGraph(int n,
                                                                        int d) {
  auto random_graph_adj = std::make_unique<std::unique_ptr<bool[]>[]>(n);
  for (auto i = 0; i < n; i++) {
    random_graph_adj[i] = std::make_unique<bool[]>(n);
  }
  for (auto i = 0; i < n; i++) {
    for (auto j = i + 1; j < n; j++) {
      random_graph_adj[i][j] =
          static_cast<double>(rand()) / RAND_MAX < static_cast<double>(d) / n;
    }
  }
  return random_graph_adj;
}

inline static int TrianglesInRandomGraph(int n, int d) {
  const auto random_graph_adj = GenRandomGraph(n, d);
  auto cnt = 0;
  for (auto i = 0; i < n; i++) {
    for (auto j = i + 1; j < n; j++) {
      for (auto k = j + 1; k < n; k++) {
        if (random_graph_adj[i][j] && random_graph_adj[i][k] &&
            random_graph_adj[j][k]) {
          cnt++;
        }
      }
    }
  }
  return cnt;
}

static const int kRepeat = 10;

int main() {
  auto begin_timepoint = std::chrono::high_resolution_clock::now();

  std::srand(std::time(nullptr));

  const int n_vals[]{1000, 100};
  const int d_vals[]{2, 3, 6};
  const auto num_of_n = sizeof(n_vals) / sizeof(int);
  const auto num_of_d = sizeof(d_vals) / sizeof(int);
  int sum[num_of_n][num_of_d]{};

  for (auto rep = 0; rep < kRepeat; rep++) {
    for (auto n_index = 0; n_index < num_of_n; n_index++) {
      for (auto d_index = 0; d_index < num_of_d; d_index++) {
        sum[n_index][d_index] +=
            TrianglesInRandomGraph(n_vals[n_index], d_vals[d_index]);
      }
    }
  }

  for (auto n_index = 0; n_index < num_of_n; n_index++) {
    for (auto d_index = 0; d_index < num_of_d; d_index++) {
      auto avg_triangles = static_cast<double>(sum[n_index][d_index]) /
                           static_cast<double>(kRepeat);
      std::cout << "n = " << n_vals[n_index] << ", d = " << d_vals[d_index]
                << ", triangles = " << avg_triangles << std::endl;
    }
  }

  std::cout << std::endl;

  auto end_timepoint = std::chrono::high_resolution_clock::now();
  auto time_elaplsed = end_timepoint - begin_timepoint;
  using namespace std::chrono_literals;
  std::cout << "time elapsed: " << time_elaplsed / 1s << "s "
            << (time_elaplsed / 1ms) % (1s / 1ms) << "ms" << std::endl;
}