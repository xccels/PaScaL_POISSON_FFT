#include "modules.hpp"
#include <cstdio>

namespace timer {

std::array<double,64> t_array;
std::array<double,64> t_array_reduce;

static std::array<double,8> t_zero;
static std::array<std::string,64> t_str;
static int ntimer;

void timer_init(int n, const std::vector<std::string>& str) {
    if (n > 64) {
        std::fprintf(stderr, "[Error] Maximun number of timer is 64\n");
        MPI_Finalize();
        std::exit(EXIT_FAILURE);
    }
    ntimer = n;
    t_array.fill(0.0);
    t_array_reduce.fill(0.0);
    t_zero.fill(0.0);
    t_str.fill("null");
    for (int i = 0; i < n && i < static_cast<int>(str.size()); ++i) {
        t_str[i] = str[i];
    }
}

void timer_stamp0(int stamp_id) {
    t_zero[stamp_id-1] = MPI_Wtime();
}

void timer_stamp(int timer_id, int stamp_id) {
    double t_curr = MPI_Wtime();
    t_array[timer_id-1] += t_curr - t_zero[stamp_id-1];
    t_zero[stamp_id-1] = t_curr;
}

void timer_start(int timer_id) {
    t_array[timer_id-1] = MPI_Wtime();
}

void timer_end(int timer_id) {
    t_array[timer_id-1] = MPI_Wtime() - t_array[timer_id-1];
}

double timer_elapsed(int timer_id) {
    return MPI_Wtime() - t_array[timer_id-1];
}

void timer_reduction() {
    MPI_Reduce(t_array.data(), t_array_reduce.data(), ntimer, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

void timer_output(int myrank, int nprocs) {
    if (myrank == 0) {
        double total_time_elapsed = 0.0;
        for (int i = 0; i < ntimer; ++i) {
            if (t_str[i] != "null") {
                total_time_elapsed += t_array_reduce[i] / nprocs;
                std::printf("[Timer] %s : (%d) : %16.9f\n", t_str[i].c_str(), i+1, t_array_reduce[i]/nprocs);
            }
        }
        printf("Total time elapsed: %16.9f\n", total_time_elapsed);
    }
}

} // namespace timer
