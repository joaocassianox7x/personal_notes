#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <omp.h>

namespace {

constexpr std::uint64_t kDefaultSamples = 400000000ULL;
constexpr std::uint64_t kSeed = 0x9E3779B97F4A7C15ULL;

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

std::uint64_t next_u64(std::uint64_t &state) {
    state ^= state >> 12U;
    state ^= state << 25U;
    state ^= state >> 27U;
    return state * 2685821657736338717ULL;
}

double uniform01(std::uint64_t &state) {
    constexpr double inv = 1.0 / 9007199254740992.0;  // 2^53
    return static_cast<double>(next_u64(state) >> 11U) * inv;
}

}  // namespace

int main(int argc, char **argv) {
    std::uint64_t samples = kDefaultSamples;
    if (argc >= 2) {
        samples = std::strtoull(argv[1], nullptr, 10);
    }

    if (argc >= 3) {
        const int requested_threads = std::atoi(argv[2]);
        if (requested_threads > 0) {
            omp_set_num_threads(requested_threads);
        }
    }

    std::uint64_t hits = 0;
    int used_threads = 1;
    const double start = omp_get_wtime();

#pragma omp parallel reduction(+ : hits)
    {
        const int thread_id = omp_get_thread_num();
        std::uint64_t state = splitmix64(kSeed + 0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(thread_id + 1));

#pragma omp single
        used_threads = omp_get_num_threads();

#pragma omp for schedule(static)
        for (std::uint64_t i = 0; i < samples; ++i) {
            const double x = uniform01(state);
            const double y = uniform01(state);
            if (x * x + y * y <= 1.0) {
                ++hits;
            }
        }
    }

    const double elapsed = omp_get_wtime() - start;
    const double integral = static_cast<double>(hits) / static_cast<double>(samples);
    const double pi_estimate = 4.0 * integral;
    const double pi_ref = std::acos(-1.0);
    const double abs_error = std::fabs(pi_estimate - pi_ref);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "threads=" << used_threads
              << " samples=" << samples
              << " integral=" << integral
              << " pi_estimate=" << pi_estimate
              << " abs_error=" << abs_error
              << " elapsed_s=" << elapsed << '\n';

    return 0;
}
