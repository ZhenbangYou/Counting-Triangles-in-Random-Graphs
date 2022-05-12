#include <immintrin.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <new>

inline static unsigned long long ReentrantRand(unsigned long long &rand_seed) {
  rand_seed = rand_seed * 1103515245U + 12345U;
  return (rand_seed >> 32) % (1U << 31);
}

inline static bool IsConnected(long long n, long long d,
                               unsigned long long &rand_seed) {
  return ReentrantRand(rand_seed) * n <= ((1U << 31) - 1) * d;
  // <=> rand() / RAND_MAX <= d / n
}

const int kAlign = 256 / 8;

inline static std::unique_ptr<std::unique_ptr<bool[]>[]> GenRandomGraph(
    int n, int d, unsigned long long &rand_seed) {
  auto random_graph_adj = std::make_unique<std::unique_ptr<bool[]>[]>(n);
  for (auto i = 0; i < n; i++) {
    random_graph_adj[i] =
        std::unique_ptr<bool[]>(new (std::align_val_t(kAlign)) bool[n]);
  }
  for (auto i = 0; i < n; i++) {
    for (auto j = i + 1; j < n; j++) {
      random_graph_adj[i][j] = IsConnected(n, d, rand_seed);
    }
  }
  return random_graph_adj;
}

const int kStride = kAlign;

inline static int TrianglesInRandomGraph(int n, int d,
                                         unsigned long long &rand_seed) {
  const auto random_graph_adj = GenRandomGraph(n, d, rand_seed);
  auto cnt = 0;
  for (auto i = 0; i < n; i++) {
    for (auto j = i + 1; j < n; j++) {
      if (random_graph_adj[i][j]) {
        auto sum = 0;

        auto k = j + 1;
        for (; k % kAlign != 0 && k < n; k++) {
          sum += random_graph_adj[i][k] & random_graph_adj[j][k];
        }

        __m256i partial_sum{};
        for (; k + kStride <= n; k += kStride) {
          auto src_ik = _mm256_load_si256(
              reinterpret_cast<const __m256i *>(&random_graph_adj[i][k]));
          auto src_jk = _mm256_load_si256(
              reinterpret_cast<const __m256i *>(&random_graph_adj[j][k]));
          partial_sum += src_ik & src_jk;
        }

        auto sum_without_remain = 0ULL;
        for (auto t = 0; t < kAlign / sizeof(long long); t++) {
          sum_without_remain += partial_sum[t];
        }

        sum_without_remain = (sum_without_remain & ((1ULL << 32) - 1)) +
                             (sum_without_remain >> 32);
        sum_without_remain = (sum_without_remain & ((1ULL << 16) - 1)) +
                             (sum_without_remain >> 16);
        sum_without_remain = (sum_without_remain & ((1ULL << 8) - 1)) +
                             (sum_without_remain >> 8);

        sum += sum_without_remain;

        for (; k < n; k++) {
          sum += random_graph_adj[i][k] & random_graph_adj[j][k];
        }
        cnt += sum;
      }
    }
  }
  return cnt;
}

static const int kRepeat = 100;

int main() {
  auto begin_timepoint = std::chrono::high_resolution_clock::now();

  const int n_vals[]{1000, 100};
  const int d_vals[]{2, 3, 6};
  const auto num_of_n = sizeof(n_vals) / sizeof(int);
  const auto num_of_d = sizeof(d_vals) / sizeof(int);
  std::atomic_int sum[num_of_n][num_of_d]{};

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
            n_vals[n_index], d_vals[d_index], rand_seed_arr[rep]);
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