//======================================================================================================================
//> @file        module_global.f90
//> @brief       This file contains a module of global input parameters for the example problem of PaScaL_TCS.
//> @details     The input parameters include global domain information, boundary conditions, fluid properties, 
//>              flow conditions, and simulation control parameters.
//> @author      
//>              - Kiha Kim (k-kiha@yonsei.ac.kr), Department of Computational Science & Engineering, Yonsei University
//>              - Ji-Hoon Kang (jhkang@kisti.re.kr), Korea Institute of Science and Technology Information
//>              - Jung-Il Choi (jic@yonsei.ac.kr), Department of Computational Science & Engineering, Yonsei University
//>
//> @date        October 2022
//> @version     1.0
//> @par         Copyright
//>              Copyright (c) 2022 Kiha Kim and Jung-Il choi, Yonsei University and 
//>              Ji-Hoon Kang, Korea Institute of Science and Technology Information, All rights reserved.
//> @par         License     
//>              This project is release under the terms of the MIT License (see LICENSE in )
//======================================================================================================================

//>
//> @brief       Module for global parameters.
//> @details     This global module has simulation parameters and a subroutine to initialize the parameters. 
//>
#include "modules.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

inline std::string trim(std::string s) {
    auto not_space = [](int ch) { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

inline bool iequals(const std::string &a, const std::string &b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                      [](char x, char y) { return std::tolower(x) == std::tolower(y); });
}

} // unnamed namespace

namespace global {

const double PI = std::acos(-1.0);

int n1, n2, n3;
int n1m, n2m, n3m;
int n1p, n2p, n3p;
int np1, np2, np3;

bool pbc1, pbc2, pbc3;
int UNIFORM1, UNIFORM2, UNIFORM3;
double GAMMA1, GAMMA2, GAMMA3;

double rms, rms_local;

double L1, L2, L3;
double H, Aspect1, Aspect2, Aspect3;
double x1_start, x2_start, x3_start;
double x1_end, x2_end, x3_end;

std::string dir_cont_fileout;
std::string dir_instantfield;
std::string dir_cont_filein;
int print_start_step, print_interval_step, print_j_index_wall, print_j_index_bulk, print_k_index;
int ContinueFilein;
int ContinueFileout;

void global_inputpara(const std::string &filename) {
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Unable to open " + filename);
    }

    std::string line;
    std::string group;

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

        auto unquote = [](const std::string &v) {
            if (v.size() >= 2 && ((v.front() == '"' && v.back() == '"') ||
                                  (v.front() == '\'' && v.back() == '\''))) {
                return v.substr(1, v.size() - 2);
            }
            return v;
        };

        std::string lkey = lower(key);
        std::string lval = lower(val);

        if (group == "meshes") {
            if (lkey == "n1m") n1m = std::stoi(val);
            else if (lkey == "n2m") n2m = std::stoi(val);
            else if (lkey == "n3m") n3m = std::stoi(val);
        } else if (group == "mpi_procs") {
            if (lkey == "np1") np1 = std::stoi(val);
            else if (lkey == "np2") np2 = std::stoi(val);
            else if (lkey == "np3") np3 = std::stoi(val);
        } else if (group == "periodic_boundary") {
            if (lkey == "pbc1") pbc1 = (lval == ".true." || lval == "true");
            else if (lkey == "pbc2") pbc2 = (lval == ".true." || lval == "true");
            else if (lkey == "pbc3") pbc3 = (lval == ".true." || lval == "true");
        } else if (group == "uniform_mesh") {
            if (lkey == "uniform1") UNIFORM1 = std::stoi(val);
            else if (lkey == "uniform2") UNIFORM2 = std::stoi(val);
            else if (lkey == "uniform3") UNIFORM3 = std::stoi(val);
        } else if (group == "mesh_stretch") {
            if (lkey == "gamma1") GAMMA1 = std::stod(val);
            else if (lkey == "gamma2") GAMMA2 = std::stod(val);
            else if (lkey == "gamma3") GAMMA3 = std::stod(val);
        } else if (group == "aspect_ratio") {
            if (lkey == "h") H = std::stod(val);
            else if (lkey == "aspect1") Aspect1 = std::stod(val);
            else if (lkey == "aspect2") Aspect2 = std::stod(val);
            else if (lkey == "aspect3") Aspect3 = std::stod(val);
        } else if (group == "sim_continue") {
            if (lkey == "continuefilein") ContinueFilein = std::stoi(val);
            else if (lkey == "continuefileout") ContinueFileout = std::stoi(val);
            else if (lkey == "dir_cont_filein") dir_cont_filein = unquote(val);
            else if (lkey == "dir_cont_fileout") dir_cont_fileout = unquote(val);
            else if (lkey == "dir_instantfield") dir_instantfield = unquote(val);
        }
    }

    n1 = n1m + 1; n1p = n1 + 1;
    n2 = n2m + 1; n2p = n2 + 1;
    n3 = n3m + 1; n3p = n3 + 1;

    L1 = H * Aspect1;
    L2 = H * Aspect2;
    L3 = H * Aspect3;

    x1_start = 0.0;
    x2_start = 0.0;
    x3_start = 0.0;

    x1_end = x1_start + L1;
    x2_end = x2_start + L2;
    x3_end = x3_start + L3;

    double tttmp[3];
    tttmp[0] = L1 / static_cast<double>(n1 - 1);
    tttmp[1] = L2 / static_cast<double>(n2 - 1);
    tttmp[2] = L3 / static_cast<double>(n3 - 1);
}

} // namespace global