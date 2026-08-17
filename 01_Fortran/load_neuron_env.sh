#!/bin/bash
# =============================================================================
# Neuron(KISTI) 클러스터용 빌드/실행 환경.
# MPMC 클러스터용 load_mpmc_env.sh 를 이 사이트에 맞춰 옮긴 것.
#
#   source 01_Fortran/load_neuron_env.sh
#
# MPMC 와 다른 점:
#   - NVHPC 25.11 (CUDA 12.9) + 번들 HPC-X OpenMPI. MPMC 는 23.7 + OpenMPI 3.1.5.
#   - CMake 는 /usr/bin/cmake 3.26.5 가 그대로 쓸 수 있어 탐색 로직이 필요 없다.
#   - GPU 는 V100(cc70) 과 A100(cc80) 파티션을 모두 쓰므로 두 아키텍처를 함께 넣는다.
#     실행 GPU 의 아키텍처가 빌드 목록에 없으면 PTX JIT 로만 돌다가 죽는다.
# =============================================================================

if ! command -v module >/dev/null 2>&1; then
    source /apps/Modules/init/bash 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null
fi
command -v module >/dev/null 2>&1 || {
    echo "[ERROR] 'module' 명령을 찾을 수 없습니다." >&2
    return 1 2>/dev/null || exit 1
}

module purge >/dev/null 2>&1
module load nvhpc/25.11_cuda12 >/dev/null 2>&1 || {
    echo "[ERROR] nvhpc/25.11_cuda12 로드 실패" >&2
    return 1 2>/dev/null || exit 1
}

: "${NVHPC_ROOT:?nvhpc 모듈이 NVHPC_ROOT 를 설정하지 않았습니다}"
export NVHPC_SDK_ROOT="${NVHPC_ROOT}"
export NVHPC_COMPILER_ROOT="${NVHPC_ROOT}/compilers"
export NVHPC_MPI_ROOT="${HPCX_MPI_DIR}"

export CUDA_VERSION=12.9
export NEURON_CUDA_HOME="${NVHPC_ROOT}/cuda/${CUDA_VERSION}"
export CUDA_HOME="${NEURON_CUDA_HOME}"
export CUDA_PATH="${NEURON_CUDA_HOME}"
export NVHPC_CUDA_HOME="${NEURON_CUDA_HOME}"
export NVCOMPILER_CUDA_HOME="${NEURON_CUDA_HOME}"

export CUDECOMP_NCCL_HOME="${NVHPC_ROOT}/comm_libs/${CUDA_VERSION}/nccl"
export NVCOMPILER_MATH_LIBS_HOME="${NVHPC_ROOT}/math_libs/${CUDA_VERSION}"

# nvcc 의 호스트 C++ 컴파일러. CUDA 12.9 는 GCC 14 까지 지원하므로 시스템 11.5 로 충분.
export NEURON_CUDA_HOST_CXX=/usr/bin/g++
export CUDAHOSTCXX="${NEURON_CUDA_HOST_CXX}"

# 빌드 대상 GPU 아키텍처
export GPU_CC_LIST="70;80"                              # cuDecomp(CMake) 용 표기
export GPU_ARCH_FLAGS="-gpu=cc70,cc80,cuda${CUDA_VERSION}"   # nvfortran 용 표기

export CMAKE_BIN=/usr/bin/cmake

# '-cuda' 링크가 -lcublas / -lcufft / -lnccl 를 찾도록 링크·런타임 경로 추가.
for _d in "${NVCOMPILER_MATH_LIBS_HOME}/targets/x86_64-linux/lib" \
          "${NEURON_CUDA_HOME}/targets/x86_64-linux/lib" \
          "${CUDECOMP_NCCL_HOME}/lib"; do
    [ -d "${_d}" ] || continue
    case ":${LIBRARY_PATH:-}:"    in *":${_d}:"*) ;; *) export LIBRARY_PATH="${_d}:${LIBRARY_PATH:-}";;    esac
    case ":${LD_LIBRARY_PATH:-}:" in *":${_d}:"*) ;; *) export LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}";; esac
done
unset _d

export PATH="${NEURON_CUDA_HOME}/bin:${PATH}"
hash -r

if [ "${NEURON_ENV_QUIET:-0}" != "1" ]; then
    echo "========================================"
    echo "Neuron NVHPC environment"
    echo "========================================"
    echo "nvfortran : $(command -v nvfortran)"
    echo "mpif90    : $(command -v mpif90)"
    echo "CUDA      : ${NVHPC_CUDA_HOME} (${CUDA_VERSION})"
    echo "NCCL      : ${CUDECOMP_NCCL_HOME}"
    echo "GPU flags : ${GPU_ARCH_FLAGS}"
    echo "cmake     : $(${CMAKE_BIN} --version | head -1)"
fi
