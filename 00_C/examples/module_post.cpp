//>
//> @brief       Module for post-treatment for Poisson equation.
//>
#include "modules.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <complex>
#include <fftw3.h>


namespace mpi_post {

bool FirstPrint1 = true, FirstPrint2 = true, FirstPrint3 = true;

MPI_Datatype ddtype_inner_domain;
MPI_Datatype ddtype_global_domain_pr_IO;
MPI_Datatype ddtype_global_domain_MPI_IO;
MPI_Datatype ddtype_global_domain_MPI_IO_aggregation;
MPI_Datatype ddtype_aggregation_chunk;

std::vector<int> cnts_pr, disp_pr;
std::vector<int> cnts_aggr, disp_aggr;

comm_IO comm_IO_aggregation;
comm_IO comm_IO_master;

std::string xyzrank;

namespace {
inline std::size_t idx(int i, int j, int k) {
    using namespace mpi_subdomain;
    return static_cast<std::size_t>(((k + 1) * (n2msub + 2) + (j + 1)) * (n1msub + 2) + (i + 1));
}
}

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
                             int kend) {
    std::ofstream realFile(filename);
    for (int j = jsta; j <= jend; ++j) {
        realFile << "# y = " << j << "\n";
        for (int k = ksta; k <= kend; ++k) {
            for (int i = ista; i <= iend; ++i) {
                int idx = k * (y_size * x_size) + j * x_size + i;
                realFile << std::setw(17) << solution[idx] << ' ';
            }
            realFile << '\n';
        }
        realFile << '\n';
    }
}

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
                                 int kend) {
    std::ofstream realFile(real_filename);
    std::ofstream imagFile(imag_filename);

    for (int j = jsta; j <= jend; ++j) {
        realFile << "# y = " << j << "\n";
        imagFile << "# y = " << j << "\n";

        for (int k = ksta; k <= kend; ++k) {
            for (int i = ista; i <= iend; ++i) {
                int idx = k * (y_size * x_size) + j * (x_size) + i;
                realFile << std::setw(12) << complex_array[idx].real() << " ";
                imagFile << std::setw(12) << complex_array[idx].imag() << " ";
            }
            realFile << "\n";
            imagFile << "\n";
        }

        realFile << "\n";
        imagFile << "\n";
    }

    realFile.close();
    imagFile.close();
}

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
                                 int kend) {
    std::ofstream realFile(real_filename);
    std::ofstream imagFile(imag_filename);

    for (int j = jsta; j <= jend; ++j) {
        realFile << "# y = " << j << "\n";
        imagFile << "# y = " << j << "\n";

        for (int k = ksta; k <= kend; ++k) {
            for (int i = ista; i <= iend; ++i) {
                int idx = (j * x_size + i) * z_size + k;
                realFile << std::setw(12) << complex_array[idx].real() << " ";
                imagFile << std::setw(12) << complex_array[idx].imag() << " ";
            }
            realFile << "\n";
            imagFile << "\n";
        }

        realFile << "\n";
        imagFile << "\n";
    }

    realFile.close();
    imagFile.close();
}

void mpi_Post_error(int myrank,
                    const std::vector<double>& Pin,
                    const std::vector<double>& exact_sol_in,
                    double &rms) {
    using namespace mpi_subdomain;
    using namespace global;
    double rms_local = 0.0;
    for (int k = 0; k < n3msub; ++k)
        for (int j = 0; j < n2msub; ++j)
            for (int i = 0; i < n1msub; ++i) {
                double diff = Pin[idx(i,j,k)] - exact_sol_in[idx(i,j,k)];
                rms_local += diff * diff;
            }
    MPI_Allreduce(&rms_local, &rms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (myrank == 0) {
        double total = std::sqrt(rms / (static_cast<double>(n1m) * n2m * n3m));
        std::printf("[Poisson] RMS = %20.10e\n", total);
    }
}

void mpi_Post_allocation(int chunk_size) {
    using namespace mpi_topology;
    using namespace mpi_subdomain;
    using namespace global;

    comm_IO_aggregation.nprocs = chunk_size;

    if (comm_1d_x3.nprocs % comm_IO_aggregation.nprocs != 0) {
        if (myrank == 0)
            std::printf("[Error] Chunk_size for IO aggregation should be a measure of mpisize in z-direction\n");
        MPI_Abort(MPI_COMM_WORLD, 11);
    }

    int color = myrank / comm_IO_aggregation.nprocs;
    MPI_Comm_split(MPI_COMM_WORLD, color, myrank, &comm_IO_aggregation.mpi_comm);
    MPI_Comm_rank(comm_IO_aggregation.mpi_comm, &comm_IO_aggregation.myrank);

    comm_IO_master.mpi_comm = MPI_COMM_NULL;
    comm_IO_master.nprocs = nprocs / comm_IO_aggregation.nprocs;
    std::vector<int> master_group_rank(comm_IO_master.nprocs);
    for (int i = 0; i < comm_IO_master.nprocs; ++i)
        master_group_rank[i] = i * comm_IO_aggregation.nprocs;

    MPI_Group mpi_world_group, mpi_master_group;
    MPI_Comm_group(MPI_COMM_WORLD, &mpi_world_group);
    MPI_Group_incl(mpi_world_group, comm_IO_master.nprocs,
                   master_group_rank.data(), &mpi_master_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, mpi_master_group, 0,
                          &comm_IO_master.mpi_comm);
    if (comm_IO_master.mpi_comm != MPI_COMM_NULL)
        MPI_Comm_rank(comm_IO_master.mpi_comm, &comm_IO_master.myrank);

    // rank string
    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << comm_1d_x1.myrank << '_'
        << std::setw(2) << std::setfill('0') << comm_1d_x2.myrank << '_'
        << std::setw(2) << std::setfill('0') << comm_1d_x3.myrank;
    xyzrank = oss.str();

    int sizes[3], subsizes[3], starts[3];

    sizes[0] = n3msub + 2; sizes[1] = n2msub + 2; sizes[2] = n1msub + 2;
    subsizes[0] = n3msub;  subsizes[1] = n2msub;  subsizes[2] = n1msub;
    starts[0] = 1;          starts[1] = 1;          starts[2] = 1;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_inner_domain);
    MPI_Type_commit(&ddtype_inner_domain);

    // post reassembly datatype
    sizes[0] = n3m; sizes[1] = n2m; sizes[2] = n1m;
    subsizes[0] = n3msub; subsizes[1] = n2msub; subsizes[2] = n1msub;
    starts[0] = 0; starts[1] = 0; starts[2] = 0;

    MPI_Datatype ddtype_temp;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_temp);
    MPI_Aint lb = 0, extent;
    int r8size;
    MPI_Type_size(MPI_DOUBLE, &r8size);
    extent = r8size;
    MPI_Type_create_resized(ddtype_temp, lb, extent,
                            &ddtype_global_domain_pr_IO);
    MPI_Type_commit(&ddtype_global_domain_pr_IO);

    std::vector<int> cart_coord(3 * nprocs);
    for (int i = 0; i < nprocs; ++i)
        MPI_Cart_coords(mpi_world_cart, i, 3, &cart_coord[3 * i]);

    std::vector<int> n1msub_cnt(comm_1d_x1.nprocs),
                     n1msub_disp(comm_1d_x1.nprocs);
    std::vector<int> n2msub_cnt(comm_1d_x2.nprocs),
                     n2msub_disp(comm_1d_x2.nprocs);
    std::vector<int> n3msub_cnt(comm_1d_x3.nprocs),
                     n3msub_disp(comm_1d_x3.nprocs);

    MPI_Allgather(&n1msub, 1, MPI_INT, n1msub_cnt.data(), 1, MPI_INT,
                  comm_1d_x1.mpi_comm);
    MPI_Allgather(&n2msub, 1, MPI_INT, n2msub_cnt.data(), 1, MPI_INT,
                  comm_1d_x2.mpi_comm);
    MPI_Allgather(&n3msub, 1, MPI_INT, n3msub_cnt.data(), 1, MPI_INT,
                  comm_1d_x3.mpi_comm);

    n1msub_disp[0] = 0;
    for (int i = 1; i < comm_1d_x1.nprocs; ++i)
        n1msub_disp[i] = n1msub_disp[i-1] + n1msub_cnt[i-1];
    n2msub_disp[0] = 0;
    for (int i = 1; i < comm_1d_x2.nprocs; ++i)
        n2msub_disp[i] = n2msub_disp[i-1] + n2msub_cnt[i-1];
    n3msub_disp[0] = 0;
    for (int i = 1; i < comm_1d_x3.nprocs; ++i)
        n3msub_disp[i] = n3msub_disp[i-1] + n3msub_cnt[i-1];

    cnts_pr.resize(nprocs, 1);
    disp_pr.resize(nprocs);
    for (int i = 0; i < nprocs; ++i) {
        int cx = cart_coord[3*i];
        int cy = cart_coord[3*i+1];
        int cz = cart_coord[3*i+2];
        disp_pr[i] = n1msub_disp[cx] + n2msub_disp[cy] * n1m
                   + n3msub_disp[cz] * n1m * n2m;
    }

    // MPI IO datatype
    sizes[0] = n3m; sizes[1] = n2m; sizes[2] = n1m;
    subsizes[0] = n3msub; subsizes[1] = n2msub; subsizes[2] = n1msub;
    starts[0] = ksta - 1; starts[1] = jsta - 1; starts[2] = ista - 1;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_global_domain_MPI_IO);
    MPI_Type_commit(&ddtype_global_domain_MPI_IO);

    // aggregation chunk
    sizes[0] = n3msub * comm_IO_aggregation.nprocs; sizes[1] = n2msub;
    sizes[2] = n1msub;
    subsizes[0] = n3msub; subsizes[1] = n2msub; subsizes[2] = n1msub;
    starts[0] = n3msub * comm_IO_aggregation.myrank;
    starts[1] = 0; starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_temp);
    MPI_Type_create_resized(ddtype_temp, lb, extent, &ddtype_aggregation_chunk);
    MPI_Type_commit(&ddtype_aggregation_chunk);

    cnts_aggr.resize(comm_IO_aggregation.nprocs, 1);
    disp_aggr.resize(comm_IO_aggregation.nprocs);
    for (int i = 0; i < comm_IO_aggregation.nprocs; ++i)
        disp_aggr[i] = n1msub * n2msub * n3msub * i;

    // aggregated MPI IO datatype
    sizes[0] = n3m; sizes[1] = n2m; sizes[2] = n1m;
    subsizes[0] = n3msub * comm_IO_aggregation.nprocs; subsizes[1] = n2msub;
    subsizes[2] = n1msub;
    starts[0] = ksta - 1 - n3msub * comm_IO_aggregation.myrank;
    starts[1] = jsta - 1; starts[2] = ista - 1;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &ddtype_global_domain_MPI_IO_aggregation);
    MPI_Type_commit(&ddtype_global_domain_MPI_IO_aggregation);
}

void mpi_Post_FileOut_InstanField(int myrank, const std::vector<double>& Pin) {
    using namespace global;
    using namespace mpi_topology;
    using namespace mpi_subdomain;

    if (comm_1d_x3.myrank == comm_1d_x3.nprocs - 1) {
        std::string fname = dir_instantfield + "Output_instantfield_XY" +
                           xyzrank + ".plt";
        FILE* fp = std::fopen(fname.c_str(), FirstPrint1 ? "w" : "a");
        if (!fp) return;
        if (FirstPrint1) {
            std::fprintf(fp, "VARIABLES=\"X\",\"Y\",\"Z\",\"P\"\n");
            std::fprintf(fp, "zone t=\"%d\" i=%d j=%d k=%d\n", 1,
                         n1sub + 1, n2sub + 1, 1);
            int k = (n3sub - 1) / 2;
            for (int j = 0; j <= n2sub; ++j)
                for (int i = 0; i <= n1sub; ++i)
                    std::fprintf(fp, "%20.10e%20.10e%20.10e%30.20e\n",
                                 x1_sub[i], x2_sub[j], x3_sub[k],
                                 Pin[idx(i, j, k)]);
            FirstPrint1 = false;
        }
        std::fclose(fp);
    }
}

void mpi_Post_FileIn_Continue_Single_Process_IO(int myrank, std::vector<double>& Pin) {
    using namespace global;
    std::string fname = dir_cont_filein + "cont_P_" + xyzrank + ".bin";
    FILE* fp = std::fopen(fname.c_str(), "rb");
    if (fp) {
        std::fread(Pin.data(), sizeof(double), Pin.size(), fp);
        std::fclose(fp);
    }
    if (myrank == 0)
        std::printf("Read continue file using single process IO\n");
}

void mpi_Post_FileOut_Continue_Single_Process_IO(int myrank, const std::vector<double>& Pin) {
    using namespace global;
    std::string fname = dir_cont_fileout + "cont_P_" + xyzrank + ".bin";
    FILE* fp = std::fopen(fname.c_str(), "wb");
    if (fp) {
        std::fwrite(Pin.data(), sizeof(double), Pin.size(), fp);
        std::fclose(fp);
    }
    if (myrank == 0)
        std::printf("Write continue file using single process IO\n");
}

void mpi_Post_FileIn_Continue_Single_Process_IO_with_Aggregation(int myrank, std::vector<double>& Pin) {
    using namespace global;
    using namespace mpi_subdomain;

    std::vector<double> var_chunk;
    if (comm_IO_aggregation.myrank == 0)
        var_chunk.resize(static_cast<std::size_t>(n1msub) * n2msub * n3msub *
                         comm_IO_aggregation.nprocs);

    if (comm_IO_aggregation.myrank == 0) {
        std::string fname = dir_cont_filein + "cont_P_" + xyzrank + ".bin";
        FILE* fp = std::fopen(fname.c_str(), "rb");
        if (fp) {
            std::fread(var_chunk.data(), sizeof(double), var_chunk.size(), fp);
            std::fclose(fp);
        }
    }

    MPI_Scatterv(var_chunk.data(), cnts_aggr.data(), disp_aggr.data(),
                 ddtype_aggregation_chunk, Pin.data(), 1, ddtype_inner_domain,
                 0, comm_IO_aggregation.mpi_comm);

    if (comm_IO_aggregation.myrank == 0)
        var_chunk.clear();

    mpi_subdomain_ghostcell_update(Pin.data());

    if (myrank == 0)
        std::printf("Read continue file using single process IO with aggregation\n");
}

void mpi_Post_FileOut_Continue_Single_Process_IO_with_Aggregation(int myrank, const std::vector<double>& Pin) {
    using namespace global;
    using namespace mpi_subdomain;

    std::vector<double> var_chunk;
    if (comm_IO_aggregation.myrank == 0)
        var_chunk.resize(static_cast<std::size_t>(n1msub) * n2msub * n3msub *
                         comm_IO_aggregation.nprocs);

    MPI_Gatherv(Pin.data(), 1, ddtype_inner_domain, var_chunk.data(),
                cnts_aggr.data(), disp_aggr.data(), ddtype_aggregation_chunk,
                0, comm_IO_aggregation.mpi_comm);

    if (comm_IO_aggregation.myrank == 0) {
        std::string fname = dir_cont_fileout + "cont_P_" + xyzrank + ".plt";
        FILE* fp = std::fopen(fname.c_str(), "wb");
        if (fp) {
            std::fwrite(var_chunk.data(), sizeof(double), var_chunk.size(), fp);
            std::fclose(fp);
        }
        var_chunk.clear();
    }

    if (myrank == 0)
        std::printf("Write continue file using single process IO with aggregation\n");
}

void mpi_Post_FileIn_Continue_MPI_IO(int myrank, std::vector<double>& Pin) {
    using namespace global;
    MPI_File filep;
    MPI_Offset disp = 0;
    MPI_File_open(MPI_COMM_WORLD, (dir_cont_filein + "cont_P.bin").c_str(),
                  MPI_MODE_RDONLY, MPI_INFO_NULL, &filep);
    MPI_File_set_view(filep, disp, MPI_DOUBLE, ddtype_global_domain_MPI_IO,
                      "native", MPI_INFO_NULL);
    MPI_File_read(filep, Pin.data(), 1, ddtype_inner_domain, MPI_STATUS_IGNORE);
    MPI_File_close(&filep);

    mpi_subdomain::mpi_subdomain_ghostcell_update(Pin.data());

    if (myrank == 0)
        std::printf("Read continue file using MPI IO\n");
}

void mpi_Post_FileOut_Continue_MPI_IO(int myrank, const std::vector<double>& Pin) {
    using namespace global;
    MPI_File filep;
    MPI_Offset disp = 0;
    MPI_File_open(MPI_COMM_WORLD, (dir_cont_fileout + "cont_P.bin").c_str(),
                  MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &filep);
    MPI_File_set_view(filep, disp, MPI_DOUBLE, ddtype_global_domain_MPI_IO,
                      "native", MPI_INFO_NULL);
    MPI_File_write(filep, Pin.data(), 1, ddtype_inner_domain, MPI_STATUS_IGNORE);
    MPI_File_close(&filep);
    if (myrank == 0)
        std::printf("Write continue file using MPI IO\n");
}

void mpi_Post_FileIn_Continue_MPI_IO_with_Aggregation(int myrank, std::vector<double>& Pin) {
    using namespace global;
    using namespace mpi_subdomain;

    std::vector<double> var_chunk;
    if (comm_IO_aggregation.myrank == 0)
        var_chunk.resize(static_cast<std::size_t>(n1msub) * n2msub * n3msub *
                         comm_IO_aggregation.nprocs);

    MPI_File filep;
    MPI_Offset disp = 0;
    if (comm_IO_aggregation.myrank == 0) {
        MPI_File_open(comm_IO_master.mpi_comm,
                      (dir_cont_filein + "cont_P.bin").c_str(), MPI_MODE_RDONLY,
                      MPI_INFO_NULL, &filep);
        MPI_File_set_view(filep, disp, MPI_DOUBLE,
                          ddtype_global_domain_MPI_IO_aggregation, "native",
                          MPI_INFO_NULL);
        MPI_File_read(filep, var_chunk.data(), var_chunk.size(), MPI_DOUBLE,
                      MPI_STATUS_IGNORE);
        MPI_File_close(&filep);
    }

    MPI_Scatterv(var_chunk.data(), cnts_aggr.data(), disp_aggr.data(),
                 ddtype_aggregation_chunk, Pin.data(), 1, ddtype_inner_domain,
                 0, comm_IO_aggregation.mpi_comm);

    if (comm_IO_aggregation.myrank == 0)
        var_chunk.clear();

    mpi_subdomain_ghostcell_update(Pin.data());

    if (myrank == 0)
        std::printf("Read continue file using MPI IO with aggregation\n");
}

void mpi_Post_FileOut_Continue_MPI_IO_with_Aggregation(int myrank, const std::vector<double>& Pin) {
    using namespace global;
    using namespace mpi_subdomain;

    std::vector<double> var_chunk;
    if (comm_IO_aggregation.myrank == 0)
        var_chunk.resize(static_cast<std::size_t>(n1msub) * n2msub * n3msub *
                         comm_IO_aggregation.nprocs);

    MPI_Gatherv(Pin.data(), 1, ddtype_inner_domain, var_chunk.data(),
                cnts_aggr.data(), disp_aggr.data(), ddtype_aggregation_chunk,
                0, comm_IO_aggregation.mpi_comm);

    MPI_File filep;
    MPI_Offset disp = 0;
    if (comm_IO_aggregation.myrank == 0) {
        MPI_File_open(comm_IO_master.mpi_comm,
                      (dir_cont_fileout + "cont_P.bin").c_str(),
                      MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &filep);
        MPI_File_set_view(filep, disp, MPI_DOUBLE,
                          ddtype_global_domain_MPI_IO_aggregation, "native",
                          MPI_INFO_NULL);
        MPI_File_write(filep, var_chunk.data(), var_chunk.size(), MPI_DOUBLE,
                       MPI_STATUS_IGNORE);
        MPI_File_close(&filep);
        var_chunk.clear();
    }

    if (myrank == 0)
        std::printf("Write continue file using MPI IO with aggregation\n");
}

void mpi_Post_FileIn_Continue_Post_Reassembly_IO(int myrank, std::vector<double>& Pin) {
    using namespace global;
    using namespace mpi_subdomain;

    std::vector<double> var_all;
    if (myrank == 0)
        var_all.resize(static_cast<std::size_t>(n1m) * n2m * n3m);

    if (myrank == 0) {
        std::string fname = dir_cont_filein + "cont_P.bin";
        FILE* fp = std::fopen(fname.c_str(), "rb");
        if (fp) {
            std::fread(var_all.data(), sizeof(double), var_all.size(), fp);
            std::fclose(fp);
        }
    }

    MPI_Scatterv(var_all.data(), cnts_pr.data(), disp_pr.data(),
                 ddtype_global_domain_pr_IO, Pin.data(), 1, ddtype_inner_domain,
                 0, MPI_COMM_WORLD);

    if (myrank == 0)
        var_all.clear();

    mpi_subdomain_ghostcell_update(Pin.data());

    if (myrank == 0)
        std::printf("Read continue file using post reassembly IO\n");
}

void mpi_Post_FileOut_Continue_Post_Reassembly_IO(int myrank, const std::vector<double>& Pin) {
    using namespace global;
    using namespace mpi_subdomain;

    std::vector<double> var_all;
    if (myrank == 0)
        var_all.resize(static_cast<std::size_t>(n1m) * n2m * n3m);

    MPI_Gatherv(Pin.data(), 1, ddtype_inner_domain, var_all.data(),
                cnts_pr.data(), disp_pr.data(), ddtype_global_domain_pr_IO, 0,
                MPI_COMM_WORLD);

    if (myrank == 0) {
        std::string fname = dir_cont_fileout + "cont_P.bin";
        FILE* fp = std::fopen(fname.c_str(), "wb");
        if (fp) {
            std::fwrite(var_all.data(), sizeof(double), var_all.size(), fp);
            std::fclose(fp);
        }
        var_all.clear();
        std::printf("Write continue file using post reassembly IO\n");
    }
}

} // namespace mpi_post