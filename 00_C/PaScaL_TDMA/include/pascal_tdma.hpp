#ifndef PASCAL_TDMA_HPP
#define PASCAL_TDMA_HPP

#include <array>
#include <string>
#include <vector>
#include <omp.h>
#include <mpi.h>

struct ptdma_plan_single {
    MPI_Comm ptdma_world;
    int n_row_rt;
    int gather_rank;
    int myrank;
    int nprocs;
    std::vector<double> A_rd, B_rd, C_rd, D_rd;
    std::vector<double> A_rt, B_rt, C_rt, D_rt;
};

struct ptdma_plan_many {
    MPI_Comm ptdma_world;
    int n_sys_rt;
    int n_row_rt;
    int nprocs;
    std::vector<MPI_Datatype> ddtype_FS;
    std::vector<int> count_send, displ_send;
    std::vector<MPI_Datatype> ddtype_BS;
    std::vector<int> count_recv, displ_recv;
    std::vector<double> A_rd, B_rd, C_rd, D_rd;
    std::vector<double> A_rt, B_rt, C_rt, D_rt;
};

namespace timer {

extern std::array<double,64> t_array;
extern std::array<double,64> t_array_reduce;

void timer_init(int n, const std::vector<std::string>& str);
void timer_stamp0(int stamp_id);
void timer_stamp(int timer_id, int stamp_id);
void timer_start(int timer_id);
void timer_end(int timer_id);
double timer_elapsed(int timer_id);
void timer_reduction();
void timer_output(int myrank, int nprocs);

} // namespace timer

void para_range(int n1, int n2, int nprocs, int myrank,
                            int* ista, int* iend);

extern "C" {

void PaScaL_TDMA_plan_single_create(ptdma_plan_single* plan, int myrank,
                                    int nprocs, MPI_Comm mpi_world,
                                    int gather_rank);
void PaScaL_TDMA_plan_single_destroy(ptdma_plan_single* plan);
void PaScaL_TDMA_single_solve(ptdma_plan_single* plan, double* A, double* B,
                              double* C, double* D, int n_row);
void PaScaL_TDMA_single_solve_cycle(ptdma_plan_single* plan, double* A,
                                    double* B, double* C, double* D,
                                    int n_row);

void PaScaL_TDMA_plan_many_create(ptdma_plan_many* plan, int n_sys,
                                  int myrank, int nprocs, MPI_Comm mpi_world);

void PaScaL_TDMA_plan_many_destroy(ptdma_plan_many* plan, int nprocs);
void PaScaL_TDMA_many_solve(ptdma_plan_many* plan, double* __restrict__ A, double* __restrict__ B,
                            double* __restrict__ C, double* __restrict__ D, int n_sys, int n_row);
void PaScaL_TDMA_many_solve_time(ptdma_plan_many* plan, double* __restrict__ A, double* __restrict__ B,
                            double* __restrict__ C, double* __restrict__ D, int n_sys, int n_row);
void PaScaL_TDMA_many_solve_cycle(ptdma_plan_many* plan, double* A,
                                  double* B, double* C, double* D,
                                  int n_sys, int n_row);

}

#endif // PASCAL_TDMA_HPP