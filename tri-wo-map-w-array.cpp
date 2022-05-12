#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <ctime>
#include <iostream>

typedef int IndexType;

static const int kRandMax = INT_MAX;

inline static int ReentrantRand(unsigned long long &rand_seed) {
  rand_seed = rand_seed * 1103515245U + 12345U;
  return (rand_seed >> 32) % (1U << 31);
}

static const int kSigmas = 4;

inline static long long TrianglesInRandomGraph(int n, int d,
                                               const int threshold_vals[],
                                               unsigned long long &rand_seed) {
  auto neighbor_upper_bound =
      d + static_cast<int>(std::ceil(std::sqrt(d) * 6));  // 6 sigma

  auto raw_mem = new IndexType[n * neighbor_upper_bound];

  auto random_graph_adj = new IndexType *[n];
  for (IndexType i = 0; i < n; i++) {
    random_graph_adj[i] = raw_mem + i * neighbor_upper_bound;
  }
  auto adj_cnt = new int[n]{};

  IndexType slice_size = n / d;
  for (IndexType i = 0; i < n; i++) {
    IndexType base = i + 1;
    for (; base + slice_size < n; base += slice_size) {
      auto rand_num = ReentrantRand(rand_seed);

      for (auto t = 0; t < kSigmas && rand_num > threshold_vals[t]; t++) {
        IndexType neighbor = ReentrantRand(rand_seed) % (slice_size - t) + base;

        // insertion sort
        auto pos = adj_cnt[i] - 1;
        for (; pos >= 0 && random_graph_adj[i][pos] > neighbor; pos--) {
          random_graph_adj[i][pos + 1] = random_graph_adj[i][pos];
        }
        random_graph_adj[i][pos + 1] = neighbor + t - (adj_cnt[i] - 1 - pos);

        adj_cnt[i]++;
      }
    }

    // remain
    auto rand_num = ReentrantRand(rand_seed);

    for (auto t = 0; t < kSigmas && rand_num > threshold_vals[t]; t++) {
      IndexType neighbor = ReentrantRand(rand_seed) % (slice_size - t) + base;

      if (neighbor >= n) {
        continue;
      }

      // insertion sort
      auto pos = adj_cnt[i] - 1;
      for (; pos >= 0 && random_graph_adj[i][pos] > neighbor; pos--) {
        random_graph_adj[i][pos + 1] = random_graph_adj[i][pos];
      }

      auto true_neighbor = neighbor + t - (adj_cnt[i] - 1 - pos);
      if (true_neighbor < n) {
        random_graph_adj[i][pos + 1] = true_neighbor;
        adj_cnt[i]++;
      }
    }
  }

  auto triangle_cnt = 0LL;
  for (IndexType i = 0; i < n; i++) {
    for (IndexType t = 0; t < adj_cnt[i]; t++) {
      auto j = random_graph_adj[i][t];
      auto iter_i = t + 1;
      auto iter_j = 0;
      while (iter_i < adj_cnt[i] && iter_j < adj_cnt[j]) {
        auto item_i = random_graph_adj[i][iter_i];
        auto item_j = random_graph_adj[j][iter_j];
        if (item_i < item_j) {
          iter_i++;
        } else if (item_i > item_j) {
          iter_j++;
        } else {
          iter_i++;
          iter_j++;
          triangle_cnt++;
        }
      }
    }
  }

  delete[] raw_mem;
  delete[] random_graph_adj;
  delete[] adj_cnt;

  return triangle_cnt;
}

inline static long long Choose(IndexType choose_from,
                               IndexType how_many_to_choose) {
  long long answer = 1;
  for (IndexType i = 0; i < how_many_to_choose; i++) {
    answer *= choose_from - i;
    answer /= (i + 1);
  }
  return answer;
}

static const int kRepeat = static_cast<int>(1e5);

int main() {
  auto begin_timepoint = std::chrono::high_resolution_clock::now();

  // const IndexType n_vals[] {1000, 100};
  // const IndexType d_vals[] {2, 3, 6};
  const std::array<IndexType, 2> n_vals{1000, 100};
  const std::array<IndexType, 3> d_vals{2, 3, 6};
  const int num_of_n = n_vals.size();
  const int num_of_d = d_vals.size();
  std::atomic_int sum[num_of_n][num_of_d]{};
  int threshold_vals[num_of_n][num_of_d][kSigmas]{};

  for (auto n_index = 0; n_index < num_of_n; n_index++) {
    for (auto d_index = 0; d_index < num_of_d; d_index++) {
      auto sum_of_prev_threshold = 0.0;
      for (int i = 0; i < kSigmas; i++) {
        auto n = n_vals[n_index];
        auto d = d_vals[d_index];
        auto ratio = static_cast<double>(d) / static_cast<double>(n);
        auto threshold = Choose(n / d, i) * std::pow(ratio, i) *
                         std::pow(1 - ratio, n / d - i);
        sum_of_prev_threshold += threshold;
        threshold_vals[n_index][d_index][i] = sum_of_prev_threshold * kRandMax;
      }
    }
  }

  auto rand_seed = time(NULL);

  unsigned long long rand_seed_arr[kRepeat]{};
  for (auto rep = 0; rep < kRepeat; rep++) {
    rand_seed_arr[rep] = rand_seed + rep;
  }

#pragma omp parallel for
  for (auto rep = 0; rep < kRepeat; rep++) {
    for (auto n_index = 0; n_index < num_of_n; n_index++) {
      for (auto d_index = 0; d_index < num_of_d; d_index++) {
        sum[n_index][d_index] += TrianglesInRandomGraph(
            n_vals[n_index], d_vals[d_index], threshold_vals[n_index][d_index],
            rand_seed_arr[rep]);
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