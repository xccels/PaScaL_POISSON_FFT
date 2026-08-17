V34 :0x24 wrapper_module
8 main.f90 S624 0
05/04/2026  22:33:37
use pascal_tdma_cuda public 0 indirect
use fft_poisson public 0 direct
use cufft public 0 indirect
use cuda_pressure public 0 direct
use mpi public 0 direct
use mpi_topology public 0 indirect
use iso_c_binding public 0 indirect
use nvf_acc_common public 0 indirect
use openacc_la public 0 direct
use nvtx public 0 indirect
use cudafor_lib_la public 0 indirect
use cudafor_la public 0 direct
use global public 0 direct
use cuda_subdomain public 0 direct
use mpi_subdomain public 0 direct
use cuda_post public 0 direct
use iso_fortran_env private
enduse
D 58 23 6 1 11 54 0 0 0 0 0
 0 54 11 11 54 54
D 61 23 6 1 11 54 0 0 0 0 0
 0 54 11 11 54 54
D 64 23 6 1 11 54 0 0 0 0 0
 0 54 11 11 54 54
D 67 23 6 1 11 54 0 0 0 0 0
 0 54 11 11 54 54
D 70 23 6 1 11 55 0 0 0 0 0
 0 55 11 11 55 55
D 73 23 6 1 11 55 0 0 0 0 0
 0 55 11 11 55 55
D 76 26 691 8 690 7
D 85 26 694 8 693 7
D 2673 26 691 8 690 7
D 2694 26 7944 8 7943 7
D 7908 26 28447 48 28446 7
D 8619 26 29469 2872 29468 7
D 8697 22 7
D 8699 22 7
D 8701 22 7
D 8703 22 7
D 8705 22 7
D 8707 22 7
D 8709 22 7
D 8711 22 7
D 8713 22 7
D 8715 22 7
D 8717 22 7
D 8719 22 7
D 9275 23 10 1 11 54 0 0 0 0 0
 0 54 11 11 54 54
S 624 24 0 0 0 9 1 0 5013 10005 0 A 0 0 0 0 B 0 1 0 0 0 0 0 0 0 0 0 0 23 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 wrapper_module
S 637 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 640 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 645 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 646 3 0 0 0 7 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7
S 647 3 0 0 0 7 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7
R 669 7 22 iso_fortran_env integer_kinds$ac
R 671 7 24 iso_fortran_env logical_kinds$ac
R 673 7 26 iso_fortran_env real_kinds$ac
R 690 25 7 iso_c_binding c_ptr
R 691 5 8 iso_c_binding val c_ptr
R 693 25 10 iso_c_binding c_funptr
R 694 5 11 iso_c_binding val c_funptr
R 728 6 45 iso_c_binding c_null_ptr$ac
R 730 6 47 iso_c_binding c_null_funptr$ac
R 731 26 48 iso_c_binding ==
R 733 26 50 iso_c_binding !=
S 811 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 48 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
R 7943 25 6 nvf_acc_common c_devptr
R 7944 5 7 nvf_acc_common cptr c_devptr
R 7950 6 13 nvf_acc_common c_null_devptr$ac
R 7988 26 51 nvf_acc_common =
S 8059 3 0 0 0 7 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 17 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7
S 8071 3 0 0 0 7 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 29 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7
R 28446 25 3 nvtx nvtxeventattributes
R 28447 5 4 nvtx version nvtxeventattributes
R 28448 5 5 nvtx size nvtxeventattributes
R 28449 5 6 nvtx category nvtxeventattributes
R 28450 5 7 nvtx colortype nvtxeventattributes
R 28451 5 8 nvtx color nvtxeventattributes
R 28452 5 9 nvtx payloadtype nvtxeventattributes
R 28453 5 10 nvtx reserved0 nvtxeventattributes
R 28454 5 11 nvtx payload nvtxeventattributes
R 28455 5 12 nvtx messagetype nvtxeventattributes
R 28456 5 13 nvtx message nvtxeventattributes
R 29468 25 1 pascal_tdma_cuda ptdma_plan_many_cuda
R 29469 5 2 pascal_tdma_cuda ptdma_world ptdma_plan_many_cuda
R 29470 5 3 pascal_tdma_cuda nprocs ptdma_plan_many_cuda
R 29471 5 4 pascal_tdma_cuda nx_sys ptdma_plan_many_cuda
R 29472 5 5 pascal_tdma_cuda ny_sys ptdma_plan_many_cuda
R 29473 5 6 pascal_tdma_cuda nz_row ptdma_plan_many_cuda
R 29474 5 7 pascal_tdma_cuda nx_sys_rt ptdma_plan_many_cuda
R 29475 5 8 pascal_tdma_cuda nz_row_rt ptdma_plan_many_cuda
R 29476 5 9 pascal_tdma_cuda nz_row_rd ptdma_plan_many_cuda
R 29480 5 13 pascal_tdma_cuda a_rd_d ptdma_plan_many_cuda
R 29481 5 14 pascal_tdma_cuda a_rd_d$sd ptdma_plan_many_cuda
R 29482 5 15 pascal_tdma_cuda a_rd_d$p ptdma_plan_many_cuda
R 29483 5 16 pascal_tdma_cuda a_rd_d$o ptdma_plan_many_cuda
R 29485 5 18 pascal_tdma_cuda b_rd_d ptdma_plan_many_cuda
R 29489 5 22 pascal_tdma_cuda b_rd_d$sd ptdma_plan_many_cuda
R 29490 5 23 pascal_tdma_cuda b_rd_d$p ptdma_plan_many_cuda
R 29491 5 24 pascal_tdma_cuda b_rd_d$o ptdma_plan_many_cuda
R 29493 5 26 pascal_tdma_cuda c_rd_d ptdma_plan_many_cuda
R 29497 5 30 pascal_tdma_cuda c_rd_d$sd ptdma_plan_many_cuda
R 29498 5 31 pascal_tdma_cuda c_rd_d$p ptdma_plan_many_cuda
R 29499 5 32 pascal_tdma_cuda c_rd_d$o ptdma_plan_many_cuda
R 29501 5 34 pascal_tdma_cuda d_rd_d ptdma_plan_many_cuda
R 29505 5 38 pascal_tdma_cuda d_rd_d$sd ptdma_plan_many_cuda
R 29506 5 39 pascal_tdma_cuda d_rd_d$p ptdma_plan_many_cuda
R 29507 5 40 pascal_tdma_cuda d_rd_d$o ptdma_plan_many_cuda
R 29512 5 45 pascal_tdma_cuda e_buff ptdma_plan_many_cuda
R 29513 5 46 pascal_tdma_cuda e_buff$sd ptdma_plan_many_cuda
R 29514 5 47 pascal_tdma_cuda e_buff$p ptdma_plan_many_cuda
R 29515 5 48 pascal_tdma_cuda e_buff$o ptdma_plan_many_cuda
R 29520 5 53 pascal_tdma_cuda a_rt_d ptdma_plan_many_cuda
R 29521 5 54 pascal_tdma_cuda a_rt_d$sd ptdma_plan_many_cuda
R 29522 5 55 pascal_tdma_cuda a_rt_d$p ptdma_plan_many_cuda
R 29523 5 56 pascal_tdma_cuda a_rt_d$o ptdma_plan_many_cuda
R 29525 5 58 pascal_tdma_cuda b_rt_d ptdma_plan_many_cuda
R 29529 5 62 pascal_tdma_cuda b_rt_d$sd ptdma_plan_many_cuda
R 29530 5 63 pascal_tdma_cuda b_rt_d$p ptdma_plan_many_cuda
R 29531 5 64 pascal_tdma_cuda b_rt_d$o ptdma_plan_many_cuda
R 29533 5 66 pascal_tdma_cuda c_rt_d ptdma_plan_many_cuda
R 29537 5 70 pascal_tdma_cuda c_rt_d$sd ptdma_plan_many_cuda
R 29538 5 71 pascal_tdma_cuda c_rt_d$p ptdma_plan_many_cuda
R 29539 5 72 pascal_tdma_cuda c_rt_d$o ptdma_plan_many_cuda
R 29541 5 74 pascal_tdma_cuda d_rt_d ptdma_plan_many_cuda
R 29545 5 78 pascal_tdma_cuda d_rt_d$sd ptdma_plan_many_cuda
R 29546 5 79 pascal_tdma_cuda d_rt_d$p ptdma_plan_many_cuda
R 29547 5 80 pascal_tdma_cuda d_rt_d$o ptdma_plan_many_cuda
R 29549 5 82 pascal_tdma_cuda e_rt_d ptdma_plan_many_cuda
R 29553 5 86 pascal_tdma_cuda e_rt_d$sd ptdma_plan_many_cuda
R 29554 5 87 pascal_tdma_cuda e_rt_d$p ptdma_plan_many_cuda
R 29555 5 88 pascal_tdma_cuda e_rt_d$o ptdma_plan_many_cuda
R 29558 5 91 pascal_tdma_cuda sendbuf ptdma_plan_many_cuda
R 29559 5 92 pascal_tdma_cuda sendbuf$sd ptdma_plan_many_cuda
R 29560 5 93 pascal_tdma_cuda sendbuf$p ptdma_plan_many_cuda
R 29561 5 94 pascal_tdma_cuda sendbuf$o ptdma_plan_many_cuda
R 29563 5 96 pascal_tdma_cuda recvbuf ptdma_plan_many_cuda
R 29565 5 98 pascal_tdma_cuda recvbuf$sd ptdma_plan_many_cuda
R 29566 5 99 pascal_tdma_cuda recvbuf$p ptdma_plan_many_cuda
R 29567 5 100 pascal_tdma_cuda recvbuf$o ptdma_plan_many_cuda
R 29569 5 102 pascal_tdma_cuda threads ptdma_plan_many_cuda
R 29570 5 103 pascal_tdma_cuda blocks ptdma_plan_many_cuda
R 29571 5 104 pascal_tdma_cuda blocks_rt ptdma_plan_many_cuda
R 29572 5 105 pascal_tdma_cuda blocks_alltoall ptdma_plan_many_cuda
R 29573 5 106 pascal_tdma_cuda shared_buffer_size ptdma_plan_many_cuda
S 30749 7 4 0 4 9275 1 624 205701 800004 100 A 0 0 0 0 B 0 15 0 0 0 0 0 0 0 0 0 0 30750 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 timer
S 30750 11 0 0 4 9 30683 624 205707 40800000 805000 A 0 0 0 0 B 0 23 0 0 0 32 0 0 30749 30749 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 _wrapper_module$2
A 13 2 0 0 0 6 637 0 0 0 13 0 0 0 0 0 0 0 0 0 0 0
A 30 2 0 0 0 6 640 0 0 0 30 0 0 0 0 0 0 0 0 0 0 0
A 32 2 0 0 0 6 645 0 0 0 32 0 0 0 0 0 0 0 0 0 0 0
A 54 2 0 0 0 7 646 0 0 0 54 0 0 0 0 0 0 0 0 0 0 0
A 55 2 0 0 0 7 647 0 0 0 55 0 0 0 0 0 0 0 0 0 0 0
A 61 1 0 1 0 58 669 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 67 1 0 1 0 64 671 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 71 1 0 3 0 70 673 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 125 1 0 0 0 76 728 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 128 1 0 0 0 85 730 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 278 2 0 0 26 6 811 0 0 0 278 0 0 0 0 0 0 0 0 0 0 0
A 1360 1 0 0 0 2694 7950 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 1523 2 0 0 809 7 8059 0 0 0 1523 0 0 0 0 0 0 0 0 0 0 0
A 1823 2 0 0 0 7 8071 0 0 0 1823 0 0 0 0 0 0 0 0 0 0 0
Z
J 69 1 1
V 61 58 7 0
R 0 61 0 0
A 0 6 0 0 1 3 1
A 0 6 0 0 1 30 1
A 0 6 0 0 1 32 1
A 0 6 0 0 1 13 0
J 71 1 1
V 67 64 7 0
R 0 67 0 0
A 0 6 0 0 1 3 1
A 0 6 0 0 1 30 1
A 0 6 0 0 1 32 1
A 0 6 0 0 1 13 0
J 73 1 1
V 71 70 7 0
R 0 73 0 0
A 0 6 0 0 1 32 1
A 0 6 0 0 1 13 0
J 133 1 1
V 125 76 7 0
S 0 76 0 0 0
A 0 6 0 0 1 2 0
J 134 1 1
V 128 85 7 0
S 0 85 0 0 0
A 0 6 0 0 1 2 0
J 36 1 1
V 1360 2694 7 0
S 0 2694 0 0 0
A 0 2673 0 0 1 125 0
T 28446 7908 0 3 0 0
A 28447 6 0 0 1 3 1
A 28448 6 0 0 1 278 1
A 28449 6 0 0 1 2 1
A 28450 6 0 0 1 3 1
A 28452 6 0 0 1 2 1
A 28455 6 0 0 1 3 0
T 29468 8619 0 0 0 0
A 29482 7 8697 0 1 2 1
A 29481 7 0 1823 1 10 1
A 29490 7 8699 0 1 2 1
A 29489 7 0 1823 1 10 1
A 29498 7 8701 0 1 2 1
A 29497 7 0 1823 1 10 1
A 29506 7 8703 0 1 2 1
A 29505 7 0 1823 1 10 1
A 29514 7 8705 0 1 2 1
A 29513 7 0 1823 1 10 1
A 29522 7 8707 0 1 2 1
A 29521 7 0 1823 1 10 1
A 29530 7 8709 0 1 2 1
A 29529 7 0 1823 1 10 1
A 29538 7 8711 0 1 2 1
A 29537 7 0 1823 1 10 1
A 29546 7 8713 0 1 2 1
A 29545 7 0 1823 1 10 1
A 29554 7 8715 0 1 2 1
A 29553 7 0 1823 1 10 1
A 29560 7 8717 0 1 2 1
A 29559 7 0 1523 1 10 1
A 29566 7 8719 0 1 2 1
A 29565 7 0 1523 1 10 0
Z
