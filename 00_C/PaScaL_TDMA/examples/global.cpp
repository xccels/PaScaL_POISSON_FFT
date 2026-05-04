//======================================================================================================================
//> @file        global.f90
//> @brief       This file contains a module of global parameters for the example problem of PaScaL_TDMA.
//> @details     The target example problem is the three-dimensional(3D) time-dependent heat conduction problem 
//>              in a unit cube domain applied with the boundary conditions of vertically constant temperature 
//>              and horizontally periodic boundaries.
//>
//> @author      
//>              - Ki-Ha Kim (k-kiha@yonsei.ac.kr), School of Mathematics and Computing (Computational Science and Engineering), Yonsei University
//>              - Ji-Hoon Kang (jhkang@kisti.re.kr), Korea Institute of Science and Technology Information
//>              - Jung-Il Choi (jic@yonsei.ac.kr), School of Mathematics and Computing (Computational Science and Engineering), Yonsei University
//>
//> @date        May 2023
//> @version     2.0
//> @par         Copyright
//>              Copyright (c) 2019-2023 Ki-Ha Kim and Jung-Il choi, Yonsei University and 
//>              Ji-Hoon Kang, Korea Institute of Science and Technology Information, All rights reserved.
//> @par         License     
//>              This project is release under the terms of the MIT License (see LICENSE file).
//======================================================================================================================

//>
//> @brief       Module for global parameters.
//> @details     This global module has simulation parameters and a subroutine to initialize the parameters. 
//>



#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <mpi.h>
#include "modules.hpp"
#include "pascal_tdma.hpp"

namespace {
inline std::string trim(std::string s) {
    auto not_space = [](int ch) { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}
}

namespace global {

const double PI = std::acos(-1.0);

double Pr;
double Ra;
int Tmax;
int nx, ny, nz;
int nxm, nym, nzm;
int nxp, nyp, nzp;
double dt, dtStart, tStart;
double lx, ly, lz;
double dx, dy, dz;
double theta_cold, theta_hot, alphaG, nu, Ct;
#ifdef _CUDA
int thread_in_x, thread_in_y, thread_in_z;
int thread_in_x_pascal, thread_in_y_pascal;
#endif

void global_inputpara(int np_dim[3], int argc, char **argv) {
    int myrank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc != 2) {
        if (myrank == 0) {
            std::cerr << "Input file name is not defined. Usage:\"mpirun -np number exe_file input_file\"\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::ifstream in(argv[1]);
    if (!in) {
        if (myrank == 0) {
            std::cerr << "Could not open " << argv[1] << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::string line;
    std::string group;
    int npx = 0, npy = 0, npz = 0;

    while (std::getline(in, line)) {
        auto pos_comment = line.find_first_of("!#");
        if (pos_comment != std::string::npos) line.erase(pos_comment);
        line = trim(line);
        if (line.empty()) continue;

        if (line[0] == '&') {
            group = trim(line.substr(1));
            std::transform(group.begin(), group.end(), group.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            continue;
        }
        if (line[0] == '/') {
            group.clear();
            continue;
        }

        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        std::string key = trim(line.substr(0, pos));
        std::string val = trim(line.substr(pos + 1));
        if (!val.empty() && val.back() == ',') val.pop_back();

        auto lower = [](std::string s) {
            std::transform(s.begin(), s.end(), s.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            return s;
        };

        std::string lkey = lower(key);

        if (group == "meshes") {
            if (lkey == "nx") nx = std::stoi(val);
            else if (lkey == "ny") ny = std::stoi(val);
            else if (lkey == "nz") nz = std::stoi(val);
        } else if (group == "procs") {
            if (lkey == "npx") npx = std::stoi(val);
            else if (lkey == "npy") npy = std::stoi(val);
            else if (lkey == "npz") npz = std::stoi(val);
        } else if (group == "time") {
            if (lkey == "tmax") Tmax = std::stoi(val);
#ifdef _CUDA
        } else if (group == "threads") {
            if (lkey == "thread_in_x") thread_in_x = std::stoi(val);
            else if (lkey == "thread_in_y") thread_in_y = std::stoi(val);
            else if (lkey == "thread_in_z") thread_in_z = std::stoi(val);
            else if (lkey == "thread_in_x_pascal") thread_in_x_pascal = std::stoi(val);
            else if (lkey == "thread_in_y_pascal") thread_in_y_pascal = std::stoi(val);
#endif
        }
    }

    np_dim[0] = npx;
    np_dim[1] = npy;
    np_dim[2] = npz;

    Pr = 5.0; Ra = 2.0e2;

    nx += 1; ny += 1; nz += 1;
    nxm = nx - 1; nym = ny - 1; nzm = nz - 1;
    nxp = nx + 1; nyp = ny + 1; nzp = nz + 1;

    dtStart = 5.0e-3; tStart = 0.0; dt = dtStart;

    lx = 1.0; ly = 1.0; lz = 1.0;
    dx = lx / static_cast<double>(nx - 1);
    dy = ly / static_cast<double>(ny);
    dz = lz / static_cast<double>(nz - 1);

    theta_cold = -1.0; theta_hot = 2.0 + theta_cold;
    alphaG = 1.0;
    nu = 1.0 / std::sqrt(Ra/(alphaG * Pr * ly * ly * ly * (theta_hot - theta_cold)));
    Ct = nu / Pr;
}

} // namespace global