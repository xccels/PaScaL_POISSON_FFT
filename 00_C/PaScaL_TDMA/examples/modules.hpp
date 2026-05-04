#ifndef PASCAL_TDMA_EXAMPLES_MODULES_HPP
#define PASCAL_TDMA_EXAMPLES_MODULES_HPP

#include <mpi.h>
#include <vector>
#include "../src/pascal_tdma.hpp"

namespace global {
extern const double PI;
extern double Pr;
extern double Ra;
extern int Tmax;
extern int nx, ny, nz;
extern int nxm, nym, nzm;
extern int nxp, nyp, nzp;
extern double dt, dtStart, tStart;
extern double lx, ly, lz;
extern double dx, dy, dz;
extern double theta_cold, theta_hot, alphaG, nu, Ct;
#ifdef _CUDA
extern int thread_in_x, thread_in_y, thread_in_z;
extern int thread_in_x_pascal, thread_in_y_pascal;
#endif
void global_inputpara(int np_dim[3], int argc, char **argv);
}

namespace mpi_topology {
struct cart_comm_1d {
    int myrank;
    int nprocs;
    int west_rank;
    int east_rank;
    MPI_Comm mpi_comm;
};

extern MPI_Comm mpi_world_cart;
extern int np_dim[3];
extern int period[3];
extern cart_comm_1d comm_1d_x, comm_1d_y, comm_1d_z;

void mpi_topology_make();
void mpi_topology_clean();
}

namespace mpi_subdomain {
extern int nx_sub, ny_sub, nz_sub;
extern int ista, iend, jsta, jend, ksta, kend;
extern std::vector<double> x_sub, y_sub, z_sub;
extern std::vector<double> dmx_sub, dmy_sub, dmz_sub;
extern std::vector<double> thetaBC3_sub, thetaBC4_sub;
extern std::vector<int> jmbc_index, jpbc_index;
extern MPI_Datatype ddtype_sendto_E, ddtype_recvfrom_W, ddtype_sendto_W, ddtype_recvfrom_E;
extern MPI_Datatype ddtype_sendto_N, ddtype_recvfrom_S, ddtype_sendto_S, ddtype_recvfrom_N;
extern MPI_Datatype ddtype_sendto_F, ddtype_recvfrom_B, ddtype_sendto_B, ddtype_recvfrom_F;
void mpi_subdomain_make(int nprocs_in_x, int myrank_in_x,
                        int nprocs_in_y, int myrank_in_y,
                        int nprocs_in_z, int myrank_in_z);
void mpi_subdomain_clean();
void mpi_subdomain_make_ghostcell_ddtype();
void mpi_subdomain_ghostcell_update(double* theta_sub,
                                    const mpi_topology::cart_comm_1d& comm_1d_x,
                                    const mpi_topology::cart_comm_1d& comm_1d_y,
                                    const mpi_topology::cart_comm_1d& comm_1d_z);
void mpi_subdomain_indices(int myrank_in_y, int nprocs_in_y);
void mpi_subdomain_mesh(int myrank_in_x, int myrank_in_y, int myrank_in_z,
                        int nprocs_in_x, int nprocs_in_y, int nprocs_in_z);
void mpi_subdomain_initialization(std::vector<double>& theta_sub,
                                  int myrank_in_y, int nprocs_in_y);
void mpi_subdomain_boundary(const std::vector<double>& theta_sub,
                            int myrank_in_y, int nprocs_in_y);
}

namespace solve_theta {
void solve_theta_plan_single(std::vector<double>& theta);
void solve_theta_plan_many(std::vector<double>& theta);
}

#endif