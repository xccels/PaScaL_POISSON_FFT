# PaScaL_Poisson_FFT

PaScaL_Poisson_FFT provides an efficient and scalable computational procedure to solve multi-dimensional elliptic partial differential equations. To ensure fast and accurate computation, the Fast Fourier Transform (FFT) library and PaScaL_TDMA library are used, and a newly designed communication scheme has been implemented to reduce communication overhead.

This library can be used for periodic boundary conditions and Neumann boundary conditions. The main algorithm for PaScaL_Poisson_FFT consists of the following four steps:

- (1) Selection of FFT algorithm based on boundary conditions:
        Periodic boundary conditions - FFT, Neumann boundary conditions - DCT
- (2) Transformation into an ordinary differential equation using 2D FFT:
        Through the 2D FFT, the 3D Poisson equation is transformed into a simple second-order ordinary differential equation for each horizontal wavenumber.
- (3) Solving the ordinary differential equation using TDMA:
        The transformed second-order ordinary differential equation is solved using the PaScaL_TDMA library.
- (4) Restoration of the solution from the horizontal wavenumber space to the real space:
        The solution obtained in the horizontal wavenumber space is restored to the real space using the inverse FFT.

# Parallel Computing Configuration

This project leverages a 1-to-1 mapping strategy between MPI processes and CUDA GPUs to enhance the computational efficiency of the PaScaL_poisson_fft solver. It is critical that each MPI process is paired with a dedicated GPU, facilitating efficient parallel computations.

## MPI and CUDA GPU Matching

- **Equal Number Required:** It's essential for the number of MPI cores to match the number of CUDA GPUs exactly. This ensures that each MPI process can operate on a dedicated GPU, optimizing data processing and computational speed.
- **Configuration Guidance:** Before launching simulations, verify that your computational environment is set up to provide an equal number of MPI cores and GPUs. This setup is vital for achieving the expected performance and accuracy in simulations.
- **Optimization:** The 1-to-1 mapping aims to maximize hardware utilization, enabling high performance for complex computational tasks. Make sure your job submission and environment setup reflect this requirement.

## Configuration Requirement

Ensure your setup aligns with the 1-to-1 MPI-to-GPU mapping by checking available resources and configuring your computational environment accordingly. This may involve adjusting your job submission scripts or settings to ensure each MPI process is allocated to a specific GPU.

## Performance Implications

Adhering to this configuration is crucial for optimizing the solver's performance. Incorrect setups, where the number of MPI processes does not match the number of GPUs, can lead to inefficiencies or computational errors. Always verify this alignment to ensure optimal operation of your simulations.

Given the project's existing Makefile, which already specifies how to execute the solver using MPI, users should refer to this setup for launching simulations. The crucial aspect to remember is maintaining the balance between MPI processes and GPUs for effective parallel computation.

# Authors
- Ki-Ha Kim (k-kiha@yonsei.ac.kr), School of Mathematics and Computing (Computational Science and Engineering), Yonsei University
- Mingyu Yang (yang926@yonsei.ac.kr), School of Mathematics and Computing (Computational Science and Engineering), Yonsei University
- TianTian Xu (tian0917@yonsei.ac.kr), School of Mathematics and Computing (Computational Science and Engineering), Yonsei University
- Jungwoo Kim (yasandy@yonsei.ac.kr), School of Mathematics and Computing (Computational Science and Engineering), Yonsei University
- Xiaomin Pan (sanhepanxiaomin@gmail.com), Department of Mathematics, Shanghai University
- Ji-Hoon Kang (jhkang@kisti.re.kr), Korea Institute of Science and Technology Information
- Oh-Kyoung Kwon (okkwon@kisti.re.kr), Korea Institute of Science and Technology Information
- Jung-Il Choi (jic@yonsei.ac.kr), School of Mathematics and Computing (Computational Science and Engineering), Yonsei University

# Usage
## Downloading PaScaL_Poisson_fft
The repository can be cloned as follows:

```
git clone <adress>
```
Alternatively, the source files can be downloaded through github menu 'Download ZIP'.

## Compile

### [Prerequisites](./doc/2_installation.md)
Prerequistites to compile PaScaL_Poisson_fft are as follows:
* MPI
* fortran compiler (`nvfortran` for GPU runs, NVIDIA HPC SDK 21.1 or higher)

### Compile and build
* Build PaScaL_Poisson_fft
    ```
    make lib
    ```
* Build an example problem after build PaScaL_Poisson_fft

    ```
    make example
    ```
* Build all

    ```
    make all
    ```

### Mores on compile option
The `Makefile` in root directory is to compile the source code, and is expected to work for most systems. The 'Makefile.inc' file in the root directory can be used to change the compiler (and MPI wrapper) and a few pre-defined compile options depending on compiler, execution environment and et al.

## Running the example
After building the example file, an executable binary, `examples.exe`, is built in the `run` folder. The `PARA_INPUT.dat` file in the `run` folder is a pre-defined input file, and the `examples.exe` can be executed as follows:
    ```
	mpirun -np 2 ./examples.exe
    ```

# Folder structure
* `src` : source files of PaScaL_Poisson_fft.
* `examples` : source files of an example problem for 3D Poisson equation.
* `include` : header files are created after building
* `lib` : a static library of PaScaL_Poisson_fft is are created after building
* `doc` : documentation
* `run` : an executable binary file for the example problem is created after building.

# Cite

# Contact
For questions or support, please contact Prof. Jung-Il Choi at jic@yonsei.ac.kr.