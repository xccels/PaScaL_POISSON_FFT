!=======================================================================================================================
!> @file        poisson_timer.f90
!> @brief       Computation/communication split timing for the Poisson solve.
!> @details     Wraps PaScaL_TDMA's timer module (PaScaL_TDMA/src/timer.f90) with the three
!>              quantities a scalability study needs:
!>
!>                (1) computation   time spent in local work: cuFFT, TDMA kernels, packing,
!>                                  transposes, scaling -- everything outside an MPI/NCCL call
!>                (2) communication time spent inside MPI or NCCL calls
!>                (3) total         the solve as a whole
!>
!>              The three are measured by one chain of stamps, so (1) + (2) = (3) holds by
!>              construction; (3) is accumulated through an independent stamp slot, so a mismatch
!>              would reveal an un-instrumented gap.
!>
!>              Every boundary synchronizes the device first.  GPU kernels are launched
!>              asynchronously, so without that the host would charge kernel time to whichever
!>              MPI call happens to block next, and the split would be meaningless.  The
!>              synchronization does perturb the total slightly; compare against an untimed run
!>              when the absolute number matters.
!>
!>              Enabled at run time by POISSON_TIMER=1, so an untimed build stays untouched.
!>
!> @author
!>              - Jungwoo Kim (yasandy@yonsei.ac.kr), School of Mathematics and Computing (Computational Science and Engineering), Yonsei University
!>
!> @date        August 2026
!> @version     1.0
!> @par         License
!>              This project is released under the terms of the MIT License (see LICENSE file).
!=======================================================================================================================
module poisson_timer

    use mpi
    use cudafor
    use timer, only : timer_init, timer_stamp0, timer_stamp, &
                      timer_reduction, t_array, t_array_reduce

    implicit none

    private

    !> Timer slots in PaScaL_TDMA's t_array
    integer, parameter, public :: POISSON_T_COMP  = 1     !< local computation
    integer, parameter, public :: POISSON_T_COMM  = 2     !< MPI / NCCL
    integer, parameter, public :: POISSON_T_TOTAL = 3     !< whole solve
    integer, parameter :: POISSON_T_COUNT = 3

    !> Stamp slots.  Two independent chains: one walks the compute/communication
    !> boundaries, the other brackets the solve as a whole.
    integer, parameter :: STAMP_PHASE = 1
    integer, parameter :: STAMP_TOTAL = 2

    logical, save :: timer_env_read = .false.
    logical, save :: timer_on       = .false.
    integer, save :: solve_count    = 0

    public :: poisson_timer_enabled
    public :: poisson_timer_reset
    public :: poisson_timer_solve_begin, poisson_timer_solve_end
    public :: poisson_timer_comm_enter,  poisson_timer_comm_exit
    public :: poisson_timer_report
    public :: poisson_timer_values

contains

    !>
    !> @brief   Is the timing instrumentation switched on?  Reads POISSON_TIMER once.
    !>
    logical function poisson_timer_enabled()
        implicit none

        character(len=32) :: value
        character(len=64) :: labels(64)
        integer :: status, length, ios, numeric_value

        if (.not. timer_env_read) then
            timer_on = .false.
            value = ''
            call get_environment_variable('POISSON_TIMER', value, length, status)
            if (status == 0 .and. length > 0) then
                read(value(1:length), *, iostat=ios) numeric_value
                if (ios == 0) then
                    timer_on = numeric_value /= 0
                else
                    select case(trim(adjustl(value(1:length))))
                    case('false', 'FALSE', 'off', 'OFF', 'no', 'NO')
                        timer_on = .false.
                    case default
                        timer_on = .true.
                    end select
                endif
            endif
            if (timer_on) then
                labels(:) = 'null'
                labels(POISSON_T_COMP)  = 'computation (local)'
                labels(POISSON_T_COMM)  = 'communication (MPI/NCCL)'
                labels(POISSON_T_TOTAL) = 'total solve'
                call timer_init(POISSON_T_COUNT, labels)
            endif
            timer_env_read = .true.
        endif

        poisson_timer_enabled = timer_on
    end function poisson_timer_enabled

    !>
    !> @brief   Drop everything accumulated so far, e.g. after warm-up solves.
    !>
    subroutine poisson_timer_reset()
        implicit none

        if (.not. poisson_timer_enabled()) return
        t_array(1:POISSON_T_COUNT) = 0.0d0
        solve_count = 0
    end subroutine poisson_timer_reset

    !>
    !> @brief   Open both stamp chains at the start of a solve.
    !>
    subroutine poisson_timer_solve_begin()
        implicit none
        integer :: ierr

        if (.not. poisson_timer_enabled()) return
        ierr = cudaDeviceSynchronize()
        call timer_stamp0(STAMP_PHASE)
        call timer_stamp0(STAMP_TOTAL)
    end subroutine poisson_timer_solve_begin

    !>
    !> @brief   Close both chains; the trailing interval is local computation.
    !>
    subroutine poisson_timer_solve_end()
        implicit none
        integer :: ierr

        if (.not. poisson_timer_enabled()) return
        ierr = cudaDeviceSynchronize()
        call timer_stamp(POISSON_T_COMP,  STAMP_PHASE)
        call timer_stamp(POISSON_T_TOTAL, STAMP_TOTAL)
        solve_count = solve_count + 1
    end subroutine poisson_timer_solve_end

    !>
    !> @brief   About to call MPI or NCCL: charge the interval just ended to computation.
    !>
    subroutine poisson_timer_comm_enter()
        implicit none
        integer :: ierr

        if (.not. poisson_timer_enabled()) return
        ierr = cudaDeviceSynchronize()
        call timer_stamp(POISSON_T_COMP, STAMP_PHASE)
    end subroutine poisson_timer_comm_enter

    !>
    !> @brief   Back from MPI or NCCL: charge the interval just ended to communication.
    !>
    subroutine poisson_timer_comm_exit()
        implicit none
        integer :: ierr

        if (.not. poisson_timer_enabled()) return
        ierr = cudaDeviceSynchronize()
        call timer_stamp(POISSON_T_COMM, STAMP_PHASE)
    end subroutine poisson_timer_comm_exit

    !>
    !> @brief   This rank's raw accumulators, for a caller that does its own reduction.
    !> @details Totals over all solves since the last reset, not per-solve averages.  Used by the
    !>          Python driver, which reduces with mpi4py and writes its own benchmark record.
    !>          All zero when the instrumentation is off.
    !>
    subroutine poisson_timer_values(comp, comm, total, nsolve)
        implicit none

        double precision, intent(out) :: comp, comm, total
        integer, intent(out) :: nsolve

        comp = 0.0d0; comm = 0.0d0; total = 0.0d0; nsolve = 0
        if (.not. poisson_timer_enabled()) return

        comp   = t_array(POISSON_T_COMP)
        comm   = t_array(POISSON_T_COMM)
        total  = t_array(POISSON_T_TOTAL)
        nsolve = solve_count
    end subroutine poisson_timer_values

    !>
    !> @brief   Report per-solve averages, reduced over ranks.
    !> @details Prints the rank-mean of each quantity divided by the number of timed solves, plus
    !>          the closure check (computation + communication) / total.  A value away from 1
    !>          means some part of the solve is not covered by the stamp chain.
    !>
    subroutine poisson_timer_report()
        implicit none

        double precision :: comp, comm, total, closure
        integer :: myrank, nprocs, ierr, nsolve

        if (.not. poisson_timer_enabled()) return

        call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
        call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
        call MPI_Allreduce(MPI_IN_PLACE, solve_count, 1, MPI_INTEGER, &
                           MPI_MAX, MPI_COMM_WORLD, ierr)
        nsolve = max(solve_count, 1)

        call timer_reduction()
        if (myrank /= 0) return

        comp  = t_array_reduce(POISSON_T_COMP)  / dble(nprocs) / dble(nsolve)
        comm  = t_array_reduce(POISSON_T_COMM)  / dble(nprocs) / dble(nsolve)
        total = t_array_reduce(POISSON_T_TOTAL) / dble(nprocs) / dble(nsolve)
        closure = 1.0d0
        if (total > 0.0d0) closure = (comp + comm) / total

        write(*,'(/,A)') '================ Poisson time split ================='
        write(*,'(A,I0,A)') 'Rank-mean per solve over ', nsolve, ' timed solves [seconds].'
        write(*,'(A)') 'Device is synchronized at every boundary, so the total is'
        write(*,'(A)') 'slightly above an untimed run.'
        write(*,'(A,1X,ES13.6)') '[TIME] (a) computation       ', comp
        write(*,'(A,1X,ES13.6)') '[TIME] (b) communication     ', comm
        write(*,'(A,1X,ES13.6)') '[TIME] (c) total             ', total
        write(*,'(A,1X,F8.5)')   '[TIME] closure (a+b)/c       ', closure
        if (total > 0.0d0) then
            write(*,'(A,1X,F6.2,A)') '[TIME] communication share   ', &
                100.0d0*comm/total, ' %'
        endif
        write(*,'(A,/)') '====================================================='

        ! Machine-readable line for the benchmark driver.
        write(*,'(A,I0,A,ES13.6,A,ES13.6,A,ES13.6,A,F8.5)') &
            'TIMESPLIT,', nprocs, ',', comp, ',', comm, ',', total, ',', closure
    end subroutine poisson_timer_report

end module poisson_timer
