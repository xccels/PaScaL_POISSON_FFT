module cuda_pressure
    ! use debug
    use global
    use openacc
    ! use nvtx
    use cufft
    use cudafor
    use mpi_subdomain
    use cuda_subdomain, only : n1sub, n2sub, n3sub, n1msub, n2msub, n3msub

    implicit none

    real(rp),         allocatable, dimension(:,:,:) :: P 
    real(rp), device, allocatable, dimension(:,:,:) :: P_d
    real(rp),     pointer, device, dimension(:,:,:) :: PRHS_d

contains 

    subroutine cuda_pressure_memory(action)
        implicit none
        character(len=*), intent(in) :: action
        integer :: i, j, k

        selectcase(action)
        case('allocate')
            allocate(P(0:n1sub,0:n2sub,0:n3sub))
            P(0:n1sub,0:n2sub,0:n3sub) = real(0.0, rp)

            allocate(P_d(0:n1sub,0:n2sub,0:n3sub))
            P_d(0:n1sub,0:n2sub,0:n3sub) = real(0.0, rp)
        case('clean')
            deallocate(P)
            deallocate(P_d)
        endselect

    end subroutine cuda_pressure_memory

    subroutine cuda_pressure_RHS(prhs_d)
        use cuda_subdomain,   only : thread_in_x, thread_in_y, thread_in_z, block_in_x, block_in_y, block_in_z, threads, blocks, share_size
        implicit none

        real(rp), device, dimension( 1:, 1:, 1:), intent(out)  :: prhs_d

        blocks  = dim3(block_in_x ,  block_in_y,  block_in_z)
        threads = dim3(thread_in_x, thread_in_y, thread_in_z)

        ! call nvtxStartRange("Poisson-RHS")
            call cuda_pressure_RHS_kernel<<<blocks, threads, 5*share_size>>>(prhs_d, n1msub, n2msub, n3msub)    
        ! call nvtxEndRange

    end subroutine cuda_pressure_RHS

    subroutine cuda_pressure_PRHS_memory(action)
        implicit none
        character(len=*), intent(in) :: action

        selectcase(action)
        case('allocate')

        case('clean')
            deallocate(PRHS_d)
        endselect
        
    end subroutine cuda_pressure_PRHS_memory

!     attributes(global) subroutine cuda_pressure_RHS_kernel(prhs_d, n1, n2, n3)
!         use cuda_subdomain, only : x1_d, x2_d, x3_d, dx3_d
        
!         implicit none

!         real(rp), device, dimension( 1:, 1:, 1:), intent(out)  :: prhs_d

!         integer,  value, intent(in) :: n1, n2, n3

!         integer :: i, j, k
!         integer :: im ,ip ,jm ,jp ,km ,kp
!         integer :: ium,iup,jvm,jvp,kwm,kwp

!         integer :: tim, ti, tip, tjm, tj, tjp, tkm, tk, tkp


!         i = (blockidx%x-1) * blockdim%x + threadidx%x; im=i-1; ip=i+1;
!         j = (blockidx%y-1) * blockdim%y + threadidx%y; jm=j-1; jp=j+1;
!         k = (blockidx%z-1) * blockdim%z + threadidx%z; km=k-1; kp=k+1;

!         if ((i <= n1) .and. (j <= n2) .and. (k <= n3)) then         
!             PRHS_d(i,j,k) = - cos(x1_d(i)*PI)*cos(x2_d(j)*PI)*cos(x3_d(k)*PI)*3.0*PI*PI
!         endif

!     end subroutine cuda_pressure_RHS_kernel

    !>
    !> @brief   Right-hand side of the analytic verification problem.
    !> @details The test solution is p = cos(PI*x)cos(PI*y)cos(PI*z), i.e. physical wavenumber
    !>          PI in every direction.  The coefficient is the eigenvalue of the *discrete*
    !>          second-order Laplacian, -sum 4*sin(PI*h/2)^2/h^2, not the continuous -3*PI^2.
    !>          cos*cos*cos is then an exact eigenvector of the discretised problem, so a correct
    !>          solve reproduces it to machine precision.  The continuous coefficient would leave
    !>          an O(h^2) gap (1.8e-5 at 256^3) and blur the correctness check.
    !>
    !>          One expression covers all three directions: the periodic cell-centred modes in x
    !>          and y and the Neumann cell-centred mode in z share -4*sin(PI*h/2)^2/h^2.  The
    !>          spacings come from the face coordinates, so no global grid data is needed here.
    !>
    attributes(global) subroutine cuda_pressure_RHS_kernel(prhs_d, n1, n2, n3)
        use cuda_subdomain, only : x1_d, x2_d, x3_d
        implicit none

        real(rp), device, dimension(1:,1:,1:), intent(out) :: prhs_d
        integer, value, intent(in) :: n1, n2, n3

        integer :: i, j, k
        real(rp) :: xc, yc, zc
        real(rp) :: hx, hy, hz, rhs_coef

        i = (blockidx%x-1) * blockdim%x + threadidx%x
        j = (blockidx%y-1) * blockdim%y + threadidx%y
        k = (blockidx%z-1) * blockdim%z + threadidx%z

        if ((i <= n1) .and. (j <= n2) .and. (k <= n3)) then

            ! Face coordinates -> cell-center coordinates
            xc = real(0.5,rp) * (x1_d(i) + x1_d(i+1))
            yc = real(0.5,rp) * (x2_d(j) + x2_d(j+1))
            zc = real(0.5,rp) * (x3_d(k) + x3_d(k+1))

            ! Cell sizes -> discrete Laplacian eigenvalue for wavenumber PI
            hx = x1_d(i+1) - x1_d(i)
            hy = x2_d(j+1) - x2_d(j)
            hz = x3_d(k+1) - x3_d(k)

            rhs_coef = -( real(4.0,rp)*sin(real(0.5,rp)*PI*hx)**2/(hx*hx)    &
                        + real(4.0,rp)*sin(real(0.5,rp)*PI*hy)**2/(hy*hy)    &
                        + real(4.0,rp)*sin(real(0.5,rp)*PI*hz)**2/(hz*hz) )

            prhs_d(i,j,k) = rhs_coef      &
                            * cos(PI*xc)  &
                            * cos(PI*yc)  &
                            * cos(PI*zc)
        endif

    end subroutine cuda_pressure_RHS_kernel
end module cuda_pressure