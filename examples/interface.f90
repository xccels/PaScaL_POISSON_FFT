module interface
    use global
    use mpi
    use cudafor

    implicit none

    interface allocate_and_init
        module procedure allocate_and_init_1d_real, allocate_and_init_1d_int
        module procedure allocate_and_init_2d_real, allocate_and_init_2d_int
        module procedure allocate_and_init_3d_real, allocate_and_init_3d_int
        module procedure allocate_and_init_4d_real
    end interface

    interface allocate_and_init_device
        module procedure allocate_and_init_1d_real_device, allocate_and_init_1d_int_device
        module procedure allocate_and_init_2d_real_device, allocate_and_init_2d_int_device
        module procedure allocate_and_init_3d_real_device, allocate_and_init_3d_int_device
    end interface

    contains

    ! Allocate_and_init interfaces
    subroutine allocate_and_init_1d_real(array, size)
        implicit none
        real(rp), allocatable, intent(out) :: array(:)
        integer, intent(in) :: size
    
        allocate(array(0:size))
        array(0:size) = real(0.0, rp)
    end subroutine allocate_and_init_1d_real
    
    subroutine allocate_and_init_1d_int(array, size)
        implicit none
        integer, allocatable, intent(out) :: array(:)
        integer, intent(in) :: size
    
        allocate(array(0:size))
        array(0:size) = 0
    end subroutine allocate_and_init_1d_int
    
    subroutine allocate_and_init_2d_real(array, size1, size2)
        implicit none
        real(rp), allocatable, intent(out) :: array(:,:)
        integer, intent(in) :: size1, size2
    
        allocate(array(0:size1, 0:size2))
        array(0:size1, 0:size2) = real(0.0, rp)
    end subroutine allocate_and_init_2d_real
    
    subroutine allocate_and_init_2d_int(array, size1, size2)
        implicit none
        integer, allocatable, intent(out) :: array(:,:)
        integer, intent(in) :: size1, size2
    
        allocate(array(0:size1, 0:size2))
        array(0:size1, 0:size2) = 0
    end subroutine allocate_and_init_2d_int
    
    subroutine allocate_and_init_3d_real(array, size1, size2, size3)
        implicit none
        real(rp), allocatable, intent(out) :: array(:,:,:)
        integer, intent(in) :: size1, size2, size3
    
        allocate(array(0:size1, 0:size2, 0:size3))
        array(0:size1, 0:size2, 0:size3) = real(0.0, rp)
    end subroutine allocate_and_init_3d_real
    
    subroutine allocate_and_init_3d_int(array, size1, size2, size3)
        implicit none
        integer, allocatable, intent(out) :: array(:,:,:)
        integer, intent(in) :: size1, size2, size3
    
        allocate(array(0:size1, 0:size2, 0:size3))
        array(0:size1, 0:size2, 0:size3) = 0
    end subroutine allocate_and_init_3d_int

    subroutine allocate_and_init_4d_real(array, size1, size2, size3, size4)
        implicit none
        real(rp), allocatable, intent(out) :: array(:,:,:,:)
        integer, intent(in) :: size1, size2, size3, size4
    
        allocate(array(0:size1, 0:size2, 0:size3, 1:size4))
        array(0:size1, 0:size2, 0:size3, 1:size4) = real(0.0, rp)
    end subroutine allocate_and_init_4d_real

    subroutine allocate_and_init_1d_real_device(array, size)
        implicit none
        real(rp), device, allocatable, intent(out) :: array(:)
        integer, intent(in) :: size
    
        allocate(array(0:size))
        array(0:size) = real(0.0, rp)
    end subroutine allocate_and_init_1d_real_device
    
    subroutine allocate_and_init_1d_int_device(array, size)
        implicit none
        integer, device, allocatable, intent(out) :: array(:)
        integer, intent(in) :: size
    
        allocate(array(0:size))
        array(0:size) = 0
    end subroutine allocate_and_init_1d_int_device
    
    subroutine allocate_and_init_2d_real_device(array, size1, size2)
        implicit none
        real(rp), device, allocatable, intent(out) :: array(:,:)
        integer, intent(in) :: size1, size2
    
        allocate(array(0:size1, 0:size2))
        array(0:size1, 0:size2) = real(0.0, rp)

    end subroutine allocate_and_init_2d_real_device
    
    subroutine allocate_and_init_2d_int_device(array, size1, size2)
        implicit none
        integer, device, allocatable, intent(out) :: array(:,:)
        integer, intent(in) :: size1, size2
    
        allocate(array(0:size1, 0:size2))
        array(0:size1, 0:size2) = 0
    end subroutine allocate_and_init_2d_int_device
    
    subroutine allocate_and_init_3d_real_device(array, size1, size2, size3)
        implicit none
        real(rp), device, allocatable, intent(out) :: array(:,:,:)
        integer, intent(in) :: size1, size2, size3
    
        allocate(array(0:size1, 0:size2, 0:size3))
        array(0:size1, 0:size2, 0:size3) = real(0.0, rp)
    end subroutine allocate_and_init_3d_real_device
    
    subroutine allocate_and_init_3d_int_device(array, size1, size2, size3)
        implicit none
        integer, device, allocatable, intent(out) :: array(:,:,:)
        integer, intent(in) :: size1, size2, size3
    
        allocate(array(0:size1, 0:size2, 0:size3))
        array(0:size1, 0:size2, 0:size3) = 0
    end subroutine allocate_and_init_3d_int_device

end module interface