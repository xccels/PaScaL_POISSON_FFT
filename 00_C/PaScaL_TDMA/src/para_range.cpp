//======================================================================================================================
//> @file        para_range.f90
//> @brief       The para_range function assigns the computing range to each MPI process.
//>
//> @author      
//>              - Ki-Ha Kim (k-kiha@yonsei.ac.kr), School of Mathematics and Computing (Computational Science and Engineering), Yonsei University
//>              - Ji-Hoon Kang (jhkang@kisti.re.kr), Korea Institute of Science and Technology Information
//>              - Jung-Il Choi (jic@yonsei.ac.kr), School of Mathematics and Computing (Computational Science and Engineering), Yonsei University
//>
//> @date        May 2023
//> @version     2.0
//> @par         Copyright
//>              Copyright (c) 2019 Ki-Ha Kim and Jung-Il choi, Yonsei University and 
//>              Ji-Hoon Kang, Korea Institute of Science and Technology Information, All rights reserved.
//> @par         License     
//>              This project is release under the terms of the MIT License (see LICENSE file).
//======================================================================================================================

//> @brief       Compute the indices of the assigned range for each MPI process .
//> @param       n1      First index of total range
//> @param       n2      Last index of total range
//> @param       nprocs  Number of MPI process
//> @param       myrank  Rank ID of my MPI process
//> @param       ista    First index of assigned range for myrank
//> @param       iend    Last index of assigned range for myrank

#include "pascal_tdma.hpp"
#include <algorithm>

void para_range(int n1, int n2, int nprocs, int myrank,
                            int* ista, int* iend) {
    int iwork1 = (n2 - n1 + 1) / nprocs;
    int iwork2 = (n2 - n1 + 1) % nprocs;
    *ista = myrank * iwork1 + n1 + std::min(myrank, iwork2);
    *iend = *ista + iwork1 - 1;
    if (iwork2 > myrank) ++(*iend);
}