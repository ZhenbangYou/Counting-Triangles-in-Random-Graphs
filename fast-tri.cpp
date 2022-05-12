#include <atomic>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <vector>

typedef long IndexType;

inline static unsigned long long ReentrantRand(unsigned long long &rand_seed) {
	rand_seed = rand_seed * 1103515245U + 12345U;
	return (rand_seed >> 32) % (1U << 31);
}

inline static long long TrianglesInRandomGraph(int n, int d,
											   unsigned long long &rand_seed) {
	auto random_graph_adj = std::make_unique<std::vector<IndexType>[]>(n);
	auto threshold        = ((1ULL << 31) - 1) * d / n;
	for (IndexType i = 0; i < n; i++) {
		for (auto j = i + 1; j < n; j++) {
			if (ReentrantRand(rand_seed) <= threshold) {
				random_graph_adj[i].push_back(j);
			}
		}
	}
	auto triangle_cnt = 0LL;
	for (IndexType i = 0; i < n; i++) {
		for (auto iter = random_graph_adj[i].cbegin();
			 iter != random_graph_adj[i].cend(); ++iter) {
			auto j      = *iter;
			auto iter_i = std::next(iter, 1);
			auto iter_j = random_graph_adj[j].cbegin();
			while (iter_i != random_graph_adj[i].cend()
				   && iter_j != random_graph_adj[j].cend()) {
				if (*iter_i < *iter_j) {
					++iter_i;
				} else if (*iter_i > *iter_j) {
					++iter_j;
				} else {
					++iter_i;
					++iter_j;
					++triangle_cnt;
				}
			}
		}
	}
	return triangle_cnt;
}

static const int kRepeat = static_cast<int>(1e3);

int main() {
	auto begin_timepoint = std::chrono::high_resolution_clock::now();

	const IndexType n_vals[] {1000, 100};
	const IndexType d_vals[] {2, 3, 6};
	const int       num_of_n = sizeof(n_vals) / sizeof(IndexType);
	const int       num_of_d = sizeof(d_vals) / sizeof(IndexType);
	std::atomic_int sum[num_of_n][num_of_d] {};

	auto rand_seed = time(NULL);

	unsigned long long rand_seed_arr[kRepeat] {};
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
			auto avg_triangles = static_cast<double>(sum[n_index][d_index])
								 / static_cast<double>(kRepeat);
			std::cout << "n = " << n_vals[n_index]
					  << ", d = " << d_vals[d_index]
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