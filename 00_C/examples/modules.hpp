#ifndef MODULES_HPP
#define MODULES_HPP

#include <string>
#include <vector>
#include <array>
#include <mpi.h>
#include <complex>
#include <fftw3.h>
#include "pascal_tdma.hpp"

namespace global {

extern const double PI;

extern int n1, n2, n3;
extern int n1m, n2m, n3m;
extern int n1p, n2p, n3p;
extern int np1, np2, np3;

extern bool pbc1, pbc2, pbc3;
extern int UNIFORM1, UNIFORM2, UNIFORM3;
extern double GAMMA1, GAMMA2, GAMMA3;

extern double rms, rms_local;

extern double L1, L2, L3;
extern double H, Aspect1, Aspect2, Aspect3;
extern double x1_start, x2_start, x3_start;
extern double x1_end, x2_end, x3_end;

extern std::string dir_cont_fileout;
extern std::string dir_instantfield;
extern std::string dir_cont_filein;
extern int print_start_step, print_interval_step, print_j_index_wall, print_j_index_bulk, print_k_index;
extern int ContinueFilein;
extern int ContinueFileout;

void global_inputpara(const std::string &filename);

} // namespace global

namespace mpi_topology {

extern MPI_Comm mpi_world_cart;
extern int nprocs;
extern int myrank;

struct cart_comm_1d {
    int myrank;
    int nprocs;
    int west_rank;
    int east_rank;
    MPI_Comm mpi_comm;
};

extern cart_comm_1d comm_1d_x1;
extern cart_comm_1d comm_1d_x2;
extern cart_comm_1d comm_1d_x3;
extern cart_comm_1d comm_1d_x1n2;

void mpi_topology_make();
void mpi_topology_clean();

} // namespace mpi_topology

namespace mpi_subdomain {

extern int n1sub, n2sub, n3sub;
extern int n1msub, n2msub, n3msub;
extern int ista, iend, jsta, jend, ksta, kend;

extern std::vector<double> x1_sub, x2_sub, x3_sub;
extern std::vector<double> x1_sub_half, x2_sub_half, x3_sub_half;
extern std::vector<double> PRHS;
extern std::vector<double> dx1_sub, dx2_sub, dx3_sub;
extern std::vector<double> dmx1_sub, dmx2_sub, dmx3_sub;
extern std::vector<double> fft_amj, fft_apj, fft_acj;

extern std::vector<int> iC_BC, iS_BC, jC_BC, jS_BC, kC_BC, kS_BC;
extern int i_indexS, j_indexS, k_indexS;

extern MPI_Datatype ddtype_sendto_E, ddtype_recvfrom_W, ddtype_sendto_W, ddtype_recvfrom_E;
extern MPI_Datatype ddtype_sendto_N, ddtype_recvfrom_S, ddtype_sendto_S, ddtype_recvfrom_N;
extern MPI_Datatype ddtype_sendto_F, ddtype_recvfrom_B, ddtype_sendto_B, ddtype_recvfrom_F;

extern int h1p, h3p;
extern int h1pKsub, n2mIsub, n2mKsub, h1pKsub_ista, h1pKsub_iend, n2mKsub_jsta, n2mKsub_jend;
extern int n3msub_Isub, h1psub, h1psub_Ksub, h1psub_Ksub_ista, h1psub_Ksub_iend;

extern std::vector<MPI_Datatype> ddtype_dble_C_in_C2I, ddtype_dble_I_in_C2I;
extern std::vector<MPI_Datatype> ddtype_cplx_C_in_C2I, ddtype_cplx_I_in_C2I;
extern std::vector<MPI_Datatype> ddtype_cplx_I_in_I2K, ddtype_cplx_K_in_I2K;
extern std::vector<MPI_Datatype> ddtype_cplx_C_in_C2K, ddtype_cplx_K_in_C2K;

//----------------------------------------------------------------------------//
// Modified pointers
extern std::vector<double> Am_r, Ac_r, Ap_r, Be_r, Am_c, Ac_c, Ap_c, Be_c;
extern std::vector<double> Am_r_modified, Ac_r_modified, Ap_r_modified, Be_r_modified;
extern std::vector<double> Am_c_modified, Ac_c_modified, Ap_c_modified, Be_c_modified;

extern std::vector<std::complex<double>> RHSIKhat_Kline;
extern std::vector<std::complex<double>> RHSIhat_Kline2;

extern std::vector<double> buffer_dp1, buffer_dp2;
extern std::vector<std::complex<double>> buffer_cd1, buffer_cd2;

// Plans
extern fftw_plan plan_fft_x;
extern fftw_plan plan_fft_z;
extern fftw_plan plan_ifft_x;
extern fftw_plan plan_ifft_z;

extern ptdma_plan_many plan_tdma, plan_tdma_modified;
//----------------------------------------------------------------------------//

extern std::vector<int> countsendI, countdistI, countsendK, countdistK;

void mpi_subdomain_make();
void mpi_subdomain_clean();
void mpi_subdomain_mesh();
void mpi_subdomain_indices();
void mpi_subdomain_indices_clean();
void mpi_subdomain_DDT_ghostcell();
void mpi_subdomain_DDT_transpose1();
void mpi_subdomain_DDT_transpose2();
void mpi_subdomain_ghostcell_update(double* Value_sub);
void subdomain_para_range(int nsta, int nend, int nprocs, int myrank,
                          int &index_sta, int &index_end);

} // namespace mpi_subdomain

namespace mpi_post {

extern bool FirstPrint1, FirstPrint2, FirstPrint3;

extern MPI_Datatype ddtype_inner_domain;
extern MPI_Datatype ddtype_global_domain_pr_IO;
extern MPI_Datatype ddtype_global_domain_MPI_IO;
extern MPI_Datatype ddtype_global_domain_MPI_IO_aggregation;
extern MPI_Datatype ddtype_aggregation_chunk;

extern std::vector<int> cnts_pr, disp_pr;
extern std::vector<int> cnts_aggr, disp_aggr;

struct comm_IO {
    int myrank;
    int nprocs;
    MPI_Comm mpi_comm;
};

extern comm_IO comm_IO_aggregation;
extern comm_IO comm_IO_master;

extern std::string xyzrank;

void mpi_Post_error(int myrank,
                    const std::vector<double>& Pin,
                    const std::vector<double>& exact_sol_in,
                    double &rms);

void mpi_Post_allocation(int chunk_size);

void mpi_Post_FileOut_InstanField(int myrank, const std::vector<double>& Pin);
void mpi_Post_FileIn_Continue_Single_Process_IO(int myrank, std::vector<double>& Pin);
void mpi_Post_FileOut_Continue_Single_Process_IO(int myrank, const std::vector<double>& Pin);
void mpi_Post_FileIn_Continue_Single_Process_IO_with_Aggregation(int myrank, std::vector<double>& Pin);
void mpi_Post_FileOut_Continue_Single_Process_IO_with_Aggregation(int myrank, const std::vector<double>& Pin);
void mpi_Post_FileIn_Continue_MPI_IO(int myrank, std::vector<double>& Pin);
void mpi_Post_FileOut_Continue_MPI_IO(int myrank, const std::vector<double>& Pin);
void mpi_Post_FileIn_Continue_MPI_IO_with_Aggregation(int myrank, std::vector<double>& Pin);
void mpi_Post_FileOut_Continue_MPI_IO_with_Aggregation(int myrank, const std::vector<double>& Pin);
void mpi_Post_FileIn_Continue_Post_Reassembly_IO(int myrank, std::vector<double>& Pin);
void mpi_Post_FileOut_Continue_Post_Reassembly_IO(int myrank, const std::vector<double>& Pin);
void write_scalar_components(const std::vector<double>& solution,
                             const std::string& filename,
                             int x_size,
                             int y_size,
                             int z_size,
                             int ista,
                             int iend,
                             int jsta,
                             int jend,
                             int ksta,
                             int kend);
void write_fft_result_components(const std::complex<double>* complex_array,
                                 const std::string& real_filename,
                                 const std::string& imag_filename,
                                 int x_size,
                                 int y_size,
                                 int z_size,
                                 int ista,
                                 int iend,
                                 int jsta,
                                 int jend,
                                 int ksta,
                                 int kend);
void write_fftz_result_components(const std::complex<double>* complex_array,
                                 const std::string& real_filename,
                                 const std::string& imag_filename,
                                 int x_size,
                                 int y_size,
                                 int z_size,
                                 int ista,
                                 int iend,
                                 int jsta,
                                 int jend,
                                 int ksta,
                                 int kend);
} // namespace mpi_post

namespace mpi_poisson {

extern std::vector<double> P;
extern std::vector<double> exact_sol;
extern std::vector<double> dzk, dxk1, dxk2;

void mpi_poisson_allocation();
void mpi_poisson_clean();
void mpi_poisson_wave_number();
void mpi_poisson_RHS();
void mpi_poisson_exact_sol();
void mpi_Poisson_FFT1(const std::vector<double>& dx2,
                      const std::vector<double>& dmx2);
void mpi_Poisson_FFT2(const std::vector<double>& dx2,
                      const std::vector<double>& dmx2);
void mpi_Poisson_FFT3(const std::vector<double>& dx2,
                      const std::vector<double>& dmx2);
void mpi_Poisson_FFT4(const std::vector<double>& dx2,
                      const std::vector<double>& dmx2);
void mpi_Poisson_FFT5(const std::vector<double>& dx2,
                      const std::vector<double>& dmx2);
} // namespace mpi_poisson

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

#endif // MODULES_HPP
