//======================================================================================================================
//> @file        solve_theta.f90
//> @brief       This file contains a solver subroutine for the example problem of PaScaL_TDMA.
//> @details     The target example problem is the three-dimensional time-dependent heat conduction problem 
//>              in a unit cube domain applied with the boundary conditions of vertically constant temperature 
//>              and horizontally periodic boundaries.
//>
//> @author      
//>              - Ki-Ha Kim (k-kiha@yonsei.ac.kr), School of Mathematics and Computing (Computational Science and Engineering), Yonsei University
//>              - Mingyu Yang (yang926@yonsei.ac.kr), School of Mathematics and Computing (Computational Science and Engineering), Yonsei University
//>              - Ji-Hoon Kang (jhkang@kisti.re.kr), Korea Institute of Science and Technology Information
//>              - Jung-Il Choi (jic@yonsei.ac.kr), School of Mathematics and Computing (Computational Science and Engineering), Yonsei University
//>
//> @date        May 2023
//> @version     2.0
//> @par         Copyright
//>              Copyright (c) 2019-2023 Ki-Ha Kim, Mingyu Yang and Jung-Il choi, Yonsei University and 
//>              Ji-Hoon Kang, Korea Institute of Science and Technology Information, All rights reserved.
//> @par         License     
//>              This project is release under the terms of the MIT License (see LICENSE file).
//======================================================================================================================



#include <cmath>
#include <iostream>
#include <vector>
#include <mpi.h>
#include "modules.hpp"
namespace solve_theta {

using namespace mpi_subdomain;
using namespace mpi_topology;
using namespace global;

namespace {
inline std::size_t idx(int i, int j, int k) {
    return (static_cast<std::size_t>(k) * (ny_sub + 1) + j) * (nx_sub + 1) + i;
}
inline std::size_t idx2(int a, int b, int n2) {
    return static_cast<std::size_t>(b) * n2 + a;
}
} // unnamed namespace

void solve_theta_plan_single(std::vector<double>& theta) {
    int myrank; MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    double t_curr = tStart;
    dt = dtStart;

    for (int time_step = 1; time_step <= Tmax; ++time_step) {
        t_curr += dt;
        if (myrank == 0)
            std::cout << "[Main] Current time step = " << time_step << std::endl;

        std::vector<double> rhs((nx_sub + 1) * (ny_sub + 1) * (nz_sub + 1), 0.0);

        for (int k = 1; k <= nz_sub - 1; ++k) {
            int kp = k + 1, km = k - 1;
            for (int j = 1; j <= ny_sub - 1; ++j) {
                int jp = j + 1, jm = j - 1;
                int jep = jpbc_index[j];
                int jem = jmbc_index[j];
                for (int i = 1; i <= nx_sub - 1; ++i) {
                    int ip = i + 1, im = i - 1;

                    double dedx1 = (theta[idx(i, j, k)] - theta[idx(im, j, k)]) / dmx_sub[i];
                    double dedx2 = (theta[idx(ip, j, k)] - theta[idx(i, j, k)]) / dmx_sub[ip];
                    double dedy3 = (theta[idx(i, j, k)] - theta[idx(i, jm, k)]) / dmy_sub[j];
                    double dedy4 = (theta[idx(i, jp, k)] - theta[idx(i, j, k)]) / dmy_sub[jp];
                    double dedz5 = (theta[idx(i, j, k)] - theta[idx(i, j, km)]) / dmz_sub[k];
                    double dedz6 = (theta[idx(i, j, kp)] - theta[idx(i, j, k)]) / dmz_sub[kp];

                    double viscous_e1 = (dedx2 - dedx1) / dx;
                    double viscous_e2 = (dedy4 - dedy3) / dy;
                    double viscous_e3 = (dedz6 - dedz5) / dz;
                    double viscous = 0.5 * Ct * (viscous_e1 + viscous_e2 + viscous_e3);

                    double ebc_down = 0.5 * Ct / dy / dmy_sub[j] *
                                     thetaBC3_sub[idx2(i, k, nx_sub + 1)];
                    double ebc_up   = 0.5 * Ct / dy / dmy_sub[jp] *
                                     thetaBC4_sub[idx2(i, k, nx_sub + 1)];
                    double ebc = (1.0 - jem) * ebc_down + (1.0 - jep) * ebc_up;

                    double eAPI = -0.5 * Ct / dx / dmx_sub[ip];
                    double eAMI = -0.5 * Ct / dx / dmx_sub[i];
                    double eACI =  0.5 * Ct / dx * (1.0 / dmx_sub[ip] + 1.0 / dmx_sub[i]);

                    double eAPK = -0.5 * Ct / dz / dmz_sub[kp];
                    double eAMK = -0.5 * Ct / dz / dmz_sub[k];
                    double eACK =  0.5 * Ct / dz * (1.0 / dmz_sub[kp] + 1.0 / dmz_sub[k]);

                    double eAPJ = -0.5 * Ct / dy * (1.0 / dmy_sub[jp]) * jep;
                    double eAMJ = -0.5 * Ct / dy * (1.0 / dmy_sub[j ]) * jem;
                    double eACJ =  0.5 * Ct / dy * (1.0 / dmy_sub[jp] + 1.0 / dmy_sub[j]);

                    double eRHS = eAPK * theta[idx(i, j, kp)] +
                                  eACK * theta[idx(i, j, k)] +
                                  eAMK * theta[idx(i, j, km)] +
                                  eAPJ * theta[idx(i, jp, k)] +
                                  eACJ * theta[idx(i, j, k)] +
                                  eAMJ * theta[idx(i, jm, k)] +
                                  eAPI * theta[idx(ip, j, k)] +
                                  eACI * theta[idx(i, j, k)] +
                                  eAMI * theta[idx(im, j, k)];

                    rhs[idx(i, j, k)] = theta[idx(i, j, k)] / dt + viscous + ebc -
                                        (theta[idx(i, j, k)] / dt + eRHS);
                }
            }
        }

        std::vector<double> ap(nz_sub - 1), am(nz_sub - 1),
            ac(nz_sub - 1), ad(nz_sub - 1);
        ptdma_plan_single pz_single;
        PaScaL_TDMA_plan_single_create(&pz_single, comm_1d_z.myrank,
                                       comm_1d_z.nprocs, comm_1d_z.mpi_comm, 0);
        for (int j = 1; j <= ny_sub - 1; ++j) {
            for (int i = 1; i <= nx_sub - 1; ++i) {
                for (int k = 1; k <= nz_sub - 1; ++k) {
                    int kp = k + 1;
                    ap[k - 1] = -0.5 * Ct / dz / dmz_sub[kp] * dt;
                    am[k - 1] = -0.5 * Ct / dz / dmz_sub[k ] * dt;
                    ac[k - 1] =  0.5 * Ct / dz * (1.0 / dmz_sub[kp] + 1.0 / dmz_sub[k]) * dt + 1.0;
                    ad[k - 1] = rhs[idx(i, j, k)] * dt;
                }
                PaScaL_TDMA_single_solve_cycle(&pz_single, am.data(), ac.data(),
                                                ap.data(), ad.data(), nz_sub - 1);
                for (int k = 1; k <= nz_sub - 1; ++k)
                    rhs[idx(i, j, k)] = ad[k - 1];
            }
        }
        PaScaL_TDMA_plan_single_destroy(&pz_single);

        ap.assign(ny_sub - 1, 0.0); am.assign(ny_sub - 1, 0.0);
        ac.assign(ny_sub - 1, 0.0); ad.assign(ny_sub - 1, 0.0);
        ptdma_plan_single py_single;
        PaScaL_TDMA_plan_single_create(&py_single, comm_1d_y.myrank,
                                       comm_1d_y.nprocs, comm_1d_y.mpi_comm, 0);
        for (int k = 1; k <= nz_sub - 1; ++k) {
            for (int i = 1; i <= nx_sub - 1; ++i) {
                for (int j = 1; j <= ny_sub - 1; ++j) {
                    int jp = j + 1, jm = j - 1;
                    int jep = jpbc_index[j];
                    int jem = jmbc_index[j];
                    ap[j - 1] = -0.5 * Ct / dy / dmy_sub[jp] * jep * dt;
                    am[j - 1] = -0.5 * Ct / dy / dmy_sub[j ] * jem * dt;
                    ac[j - 1] =  0.5 * Ct / dy * (1.0 / dmy_sub[jp] + 1.0 / dmy_sub[j]) * dt + 1.0;
                    ad[j - 1] = rhs[idx(i, j, k)];
                }
                PaScaL_TDMA_single_solve(&py_single, am.data(), ac.data(),
                                          ap.data(), ad.data(), ny_sub - 1);
                for (int j = 1; j <= ny_sub - 1; ++j)
                    rhs[idx(i, j, k)] = ad[j - 1];
            }
        }
        PaScaL_TDMA_plan_single_destroy(&py_single);

        ap.assign(nx_sub - 1, 0.0); am.assign(nx_sub - 1, 0.0);
        ac.assign(nx_sub - 1, 0.0); ad.assign(nx_sub - 1, 0.0);
        ptdma_plan_single px_single;
        PaScaL_TDMA_plan_single_create(&px_single, comm_1d_x.myrank,
                                       comm_1d_x.nprocs, comm_1d_x.mpi_comm, 0);
        for (int k = 1; k <= nz_sub - 1; ++k) {
            for (int j = 1; j <= ny_sub - 1; ++j) {
                for (int i = 1; i <= nx_sub - 1; ++i) {
                    int ip = i + 1, im = i - 1;
                    ap[i - 1] = -0.5 * Ct / dx / dmx_sub[ip] * dt;
                    am[i - 1] = -0.5 * Ct / dx / dmx_sub[i ] * dt;
                    ac[i - 1] =  0.5 * Ct / dx * (1.0 / dmx_sub[ip] + 1.0 / dmx_sub[i]) * dt + 1.0;
                    ad[i - 1] = rhs[idx(i, j, k)];
                }
                PaScaL_TDMA_single_solve_cycle(&px_single, am.data(), ac.data(),
                                                ap.data(), ad.data(), nx_sub - 1);
                for (int i = 1; i <= nx_sub - 1; ++i)
                    theta[idx(i, j, k)] += ad[i - 1];
            }
        }
        PaScaL_TDMA_plan_single_destroy(&px_single);

        mpi_subdomain_ghostcell_update(theta.data(), comm_1d_x,
                                       comm_1d_y, comm_1d_z);
    }
}

void solve_theta_plan_many(std::vector<double>& theta) {
    int myrank; MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    double t_curr = tStart;
    dt = dtStart;

    for (int time_step = 1; time_step <= Tmax; ++time_step) {
        t_curr += dt;
        if (myrank == 0)
            std::cout << "[Main] Current time step = " << time_step << std::endl;

        std::vector<double> rhs((nx_sub + 1) * (ny_sub + 1) * (nz_sub + 1), 0.0);

        for (int k = 1; k <= nz_sub - 1; ++k) {
            int kp = k + 1, km = k - 1;
            for (int j = 1; j <= ny_sub - 1; ++j) {
                int jp = j + 1, jm = j - 1;
                int jep = jpbc_index[j];
                int jem = jmbc_index[j];
                for (int i = 1; i <= nx_sub - 1; ++i) {
                    int ip = i + 1, im = i - 1;

                    double dedx1 = (theta[idx(i, j, k)] - theta[idx(im, j, k)]) / dmx_sub[i];
                    double dedx2 = (theta[idx(ip, j, k)] - theta[idx(i, j, k)]) / dmx_sub[ip];
                    double dedy3 = (theta[idx(i, j, k)] - theta[idx(i, jm, k)]) / dmy_sub[j];
                    double dedy4 = (theta[idx(i, jp, k)] - theta[idx(i, j, k)]) / dmy_sub[jp];
                    double dedz5 = (theta[idx(i, j, k)] - theta[idx(i, j, km)]) / dmz_sub[k];
                    double dedz6 = (theta[idx(i, j, kp)] - theta[idx(i, j, k)]) / dmz_sub[kp];

                    double viscous_e1 = (dedx2 - dedx1) / dx;
                    double viscous_e2 = (dedy4 - dedy3) / dy;
                    double viscous_e3 = (dedz6 - dedz5) / dz;
                    double viscous = 0.5 * Ct * (viscous_e1 + viscous_e2 + viscous_e3);

                    double ebc_down = 0.5 * Ct / dy / dmy_sub[j] *
                                     thetaBC3_sub[idx2(i, k, nx_sub + 1)];
                    double ebc_up   = 0.5 * Ct / dy / dmy_sub[jp] *
                                     thetaBC4_sub[idx2(i, k, nx_sub + 1)];
                    double ebc = (1.0 - jem) * ebc_down + (1.0 - jep) * ebc_up;

                    double eAPI = -0.5 * Ct / dx / dmx_sub[ip];
                    double eAMI = -0.5 * Ct / dx / dmx_sub[i];
                    double eACI =  0.5 * Ct / dx * (1.0 / dmx_sub[ip] + 1.0 / dmx_sub[i]);

                    double eAPK = -0.5 * Ct / dz / dmz_sub[kp];
                    double eAMK = -0.5 * Ct / dz / dmz_sub[k];
                    double eACK =  0.5 * Ct / dz * (1.0 / dmz_sub[kp] + 1.0 / dmz_sub[k]);

                    double eAPJ = -0.5 * Ct / dy * (1.0 / dmy_sub[jp]) * jep;
                    double eAMJ = -0.5 * Ct / dy * (1.0 / dmy_sub[j ]) * jem;
                    double eACJ =  0.5 * Ct / dy * (1.0 / dmy_sub[jp] + 1.0 / dmy_sub[j]);

                    double eRHS = eAPK * theta[idx(i, j, kp)] +
                                  eACK * theta[idx(i, j, k)] +
                                  eAMK * theta[idx(i, j, km)] +
                                  eAPJ * theta[idx(i, jp, k)] +
                                  eACJ * theta[idx(i, j, k)] +
                                  eAMJ * theta[idx(i, jm, k)] +
                                  eAPI * theta[idx(ip, j, k)] +
                                  eACI * theta[idx(i, j, k)] +
                                  eAMI * theta[idx(im, j, k)];

                    rhs[idx(i, j, k)] = theta[idx(i, j, k)] / dt + viscous + ebc -
                                        (theta[idx(i, j, k)] / dt + eRHS);
                }
            }
        }

        std::vector<double> ap((nx_sub - 1) * (nz_sub - 1));
        std::vector<double> am(ap.size()), ac(ap.size()), ad(ap.size());
        ptdma_plan_many pz_many;
        PaScaL_TDMA_plan_many_create(&pz_many, nx_sub - 1, comm_1d_z.myrank,
                                     comm_1d_z.nprocs, comm_1d_z.mpi_comm);
        for (int j = 1; j <= ny_sub - 1; ++j) {
            for (int k = 1; k <= nz_sub - 1; ++k) {
                int kp = k + 1;
                for (int i = 1; i <= nx_sub - 1; ++i) {
                    std::size_t id = idx2(i - 1, k - 1, nx_sub - 1);
                    ap[id] = -0.5 * Ct / dz / dmz_sub[kp] * dt;
                    am[id] = -0.5 * Ct / dz / dmz_sub[k ] * dt;
                    ac[id] =  0.5 * Ct / dz * (1.0 / dmz_sub[kp] + 1.0 / dmz_sub[k]) * dt + 1.0;
                    ad[id] = rhs[idx(i, j, k)] * dt;
                }
            }
            PaScaL_TDMA_many_solve_cycle(&pz_many, am.data(), ac.data(),
                                         ap.data(), ad.data(), nx_sub - 1,
                                         nz_sub - 1);
            for (int k = 1; k <= nz_sub - 1; ++k)
                for (int i = 1; i <= nx_sub - 1; ++i)
                    rhs[idx(i, j, k)] = ad[idx2(i - 1, k - 1, nx_sub - 1)];
        }
        PaScaL_TDMA_plan_many_destroy(&pz_many, comm_1d_z.nprocs);

        ap.assign((nx_sub - 1) * (ny_sub - 1), 0.0);
        am.assign(ap.size(), 0.0);
        ac.assign(ap.size(), 0.0);
        ad.assign(ap.size(), 0.0);
        ptdma_plan_many py_many;
        PaScaL_TDMA_plan_many_create(&py_many, nx_sub - 1, comm_1d_y.myrank,
                                     comm_1d_y.nprocs, comm_1d_y.mpi_comm);
        for (int k = 1; k <= nz_sub - 1; ++k) {
            for (int j = 1; j <= ny_sub - 1; ++j) {
                int jp = j + 1, jm = j - 1;
                int jep = jpbc_index[j];
                int jem = jmbc_index[j];
                for (int i = 1; i <= nx_sub - 1; ++i) {
                    std::size_t id = idx2(i - 1, j - 1, nx_sub - 1);
                    ap[id] = -0.5 * Ct / dy / dmy_sub[jp] * jep * dt;
                    am[id] = -0.5 * Ct / dy / dmy_sub[j ] * jem * dt;
                    ac[id] =  0.5 * Ct / dy * (1.0 / dmy_sub[jp] + 1.0 / dmy_sub[j]) * dt + 1.0;
                    ad[id] = rhs[idx(i, j, k)];
                }
            }
            PaScaL_TDMA_many_solve(&py_many, am.data(), ac.data(), ap.data(),
                                    ad.data(), nx_sub - 1, ny_sub - 1);
            for (int j = 1; j <= ny_sub - 1; ++j)
                for (int i = 1; i <= nx_sub - 1; ++i)
                    rhs[idx(i, j, k)] = ad[idx2(i - 1, j - 1, nx_sub - 1)];
        }
        PaScaL_TDMA_plan_many_destroy(&py_many, comm_1d_y.nprocs);

        ap.assign((ny_sub - 1) * (nx_sub - 1), 0.0);
        am.assign(ap.size(), 0.0);
        ac.assign(ap.size(), 0.0);
        ad.assign(ap.size(), 0.0);
        ptdma_plan_many px_many;
        PaScaL_TDMA_plan_many_create(&px_many, ny_sub - 1, comm_1d_x.myrank,
                                     comm_1d_x.nprocs, comm_1d_x.mpi_comm);
        for (int k = 1; k <= nz_sub - 1; ++k) {
            for (int j = 1; j <= ny_sub - 1; ++j) {
                for (int i = 1; i <= nx_sub - 1; ++i) {
                    std::size_t id = idx2(j - 1, i - 1, ny_sub - 1);
                    int ip = i + 1, im = i - 1;
                    ap[id] = -0.5 * Ct / dx / dmx_sub[ip] * dt;
                    am[id] = -0.5 * Ct / dx / dmx_sub[i ] * dt;
                    ac[id] =  0.5 * Ct / dx * (1.0 / dmx_sub[ip] + 1.0 / dmx_sub[i]) * dt + 1.0;
                    ad[id] = rhs[idx(i, j, k)];
                }
            }
            PaScaL_TDMA_many_solve_cycle(&px_many, am.data(), ac.data(),
                                         ap.data(), ad.data(), ny_sub - 1,
                                         nx_sub - 1);
            for (int j = 1; j <= ny_sub - 1; ++j)
                for (int i = 1; i <= nx_sub - 1; ++i)
                    theta[idx(i, j, k)] += ad[idx2(j - 1, i - 1, ny_sub - 1)];
        }
        PaScaL_TDMA_plan_many_destroy(&px_many, comm_1d_x.nprocs);

        mpi_subdomain_ghostcell_update(theta.data(), comm_1d_x,
                                       comm_1d_y, comm_1d_z);
    }
}

} // namespace solve_theta
