#!/usr/bin/env bash
# =============================================================================
# PaScaL_POISSON_FFT - Python(GPU) 환경 설정
#
#   사용법:  source 02_Python/setup_env.sh
#   ★ 반드시 'source' (module load / conda activate 가 현재 셸에 남아야 함)
#
#   하는 일 (모두 idempotent — 여러 번 source 해도 안전):
#     1) 01_Fortran/load_neuron_env.sh 로 툴체인 로드
#        (nvhpc 25.11 + 번들 HPC-X OpenMPI + CUDA 12.9 + NCCL)
#     2) conda env 없으면 environment.yml 로 생성 (python+pip 만 conda, 나머지는 pip wheel)
#     3) conda env 활성화
#     4) mpi4py 없으면 '모듈 MPI' 에 맞춰 소스 빌드 (conda 로 깔지 않음)
#     5) 검증 출력
#
#   툴체인 정의는 01_Fortran/load_neuron_env.sh 한 곳에만 둔다. .so 를 링크한 MPI 와
#   파이썬이 쓰는 MPI 가 갈라지면 py2f() 핸들이 통하지 않으므로, 빌드와 실행이 같은
#   환경 정의를 공유해야 한다.
#
#   conda 는 홈이 아니라 /scratch 를 쓴다. 홈의 제약은 용량(15G/64G, 여유 있음)이 아니라
#   파일 개수 쿼터(99,357/100,000)이고, conda env 하나가 파일 2만 개 수준이라 홈에는
#   들어가지 않는다. 패키지 캐시·env·pip 캐시를 모두 /scratch 로 돌린다.
#
#   조정 (환경변수로 덮어쓰기 가능):
#     ENV_NAME=... CONDA_SH=... POISSON_CONDA_ROOT=... source setup_env.sh
# =============================================================================

_poisson_setup() {
    local ENV_NAME="${ENV_NAME:-pascal-poisson-gpu}"
    local CONDA_ROOT="${POISSON_CONDA_ROOT:-/scratch/${USER}/conda}"
    local here yml cbase t sh env_prefix

    here="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )" && pwd )"
    yml="${here}/environment.yml"
    [ -f "$yml" ] || { echo "[ERR] environment.yml 없음: $yml" >&2; return 1; }

    # ---- 1) 툴체인 ----
    local envsh="${here}/../01_Fortran/load_neuron_env.sh"
    [ -f "$envsh" ] || { echo "[ERR] load_neuron_env.sh 없음: $envsh" >&2; return 1; }
    echo "==> 툴체인 로드: 01_Fortran/load_neuron_env.sh"
    NEURON_ENV_QUIET=1 source "$envsh" || { echo "[ERR] 툴체인 로드 실패" >&2; return 1; }

    echo "==> 툴체인:"
    for t in nvfortran nvcc mpicc mpifort; do
        if command -v "$t" >/dev/null 2>&1; then printf '    %-9s %s\n' "$t" "$(command -v "$t")"
        else echo "    [WARN] $t 가 PATH 에 없음"; fi
    done
    printf '    %-9s %s\n' "CUDA_PATH" "${CUDA_PATH}"

    # ---- 2,3) conda env ----
    if ! command -v conda >/dev/null 2>&1; then
        for sh in "${CONDA_SH:-}" \
                  /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh \
                  /apps/applications/Miniconda/24.11.1/etc/profile.d/conda.sh; do
            [ -n "$sh" ] && [ -f "$sh" ] && { source "$sh"; break; }
        done
    fi
    command -v conda >/dev/null 2>&1 || { echo "[ERR] conda 없음 (CONDA_SH 로 지정하세요)" >&2; return 1; }
    cbase="$(conda info --base 2>/dev/null)"
    [ -f "${cbase}/etc/profile.d/conda.sh" ] && source "${cbase}/etc/profile.d/conda.sh"

    # conda 가 홈 대신 /scratch 를 쓰도록. env 는 전체 경로로 만들고 활성화하므로,
    # 나중에 이 변수를 잊고 conda 를 써도 activate 는 그대로 동작한다.
    export CONDA_PKGS_DIRS="${CONDA_ROOT}/pkgs"
    export CONDA_ENVS_DIRS="${CONDA_ROOT}/envs"
    export PIP_CACHE_DIR="${CONDA_ROOT}/pip-cache"
    # CuPy 는 실행 중 JIT 커널을 ~/.cupy/kernel_cache 에 쓰고, matplotlib 은
    # ~/.config/matplotlib 을 만든다.  홈 파일 쿼터가 꽉 차면 이것들이
    # 'Disk quota exceeded' 로 죽으므로 캐시를 전부 /scratch 로 돌린다.
    export CUPY_CACHE_DIR="${CONDA_ROOT}/cupy-cache"
    export MPLCONFIGDIR="${CONDA_ROOT}/mpl-cache"
    mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_DIRS" "$PIP_CACHE_DIR" \
             "$CUPY_CACHE_DIR" "$MPLCONFIGDIR" \
        || { echo "[ERR] conda 작업 디렉터리 생성 실패: $CONDA_ROOT" >&2; return 1; }
    env_prefix="${CONDA_ENVS_DIRS}/${ENV_NAME}"

    if [ -d "${env_prefix}/conda-meta" ]; then
        echo "==> conda env 존재 → 생성 건너뜀: ${env_prefix}"
    else
        echo "==> conda env 생성: ${env_prefix}"
        echo "    (python+pip 만 conda, 나머지는 pip wheel)"
        echo "    (conda-forge repodata 최초 1회 다운로드로 수 분 걸릴 수 있음)"
        conda env create -f "$yml" -p "$env_prefix" \
            || { echo "[ERR] conda env 생성 실패" >&2; return 1; }
    fi

    echo "==> conda activate ${env_prefix}"
    conda activate "$env_prefix" || { echo "[ERR] conda activate 실패" >&2; return 1; }

    # ---- 4) mpi4py : 모듈 MPI 에 맞춰 소스 빌드 ----
    if python -c "import mpi4py" 2>/dev/null; then
        echo "==> mpi4py 설치됨 → 건너뜀 (MPI 모듈 변경 시: pip uninstall -y mpi4py 후 재실행)"
    else
        command -v mpicc >/dev/null 2>&1 || { echo "[ERR] mpicc 없음 — mpi4py 빌드 불가" >&2; return 1; }
        # HPC-X mpicc 의 기본 백엔드는 nvc 라서 gcc 전용 플래그(-fwrapv 등)를 거부함.
        # OMPI_CC=gcc 로 백엔드만 gcc 로 바꾸고, libmpi 는 그대로 HPC-X 를 링크한다.
        echo "==> mpi4py 소스 빌드 (mpicc 백엔드=gcc, libmpi=HPC-X)"
        OMPI_CC="${OMPI_CC:-gcc}" MPICC="$(command -v mpicc)" \
            pip install --no-binary=mpi4py mpi4py || { echo "[ERR] mpi4py 빌드 실패" >&2; return 1; }
    fi

    # ---- 5) 검증 ----
    echo "==> 검증:"
    nvidia-smi -L 2>/dev/null | sed 's/^/    /'
    POISSON_PY_DIR="$here" python - <<'PY'
import os, sys

# .so 를 mpi4py 보다 먼저 로드한다 (pascal_poisson.py 와 같은 이유).
# 반대 순서면 라이브러리 안의 OpenACC 런타임이 GPU 를 0개로 보고,
# 솔버의 첫 !$acc 영역에서 'No accelerator device found' 로 죽는다.
so = os.path.join(os.environ["POISSON_PY_DIR"], "lib", "libpoisson_capi.so")
if os.path.exists(so):
    import ctypes
    lib = ctypes.CDLL(so)
    a, b = ctypes.c_int(0), ctypes.c_int(0)
    lib.poisson_para_range.argtypes = [ctypes.c_int]*4 + [ctypes.POINTER(ctypes.c_int)]*2
    lib.poisson_para_range(1, 256, 2, 0, ctypes.byref(a), ctypes.byref(b))
    ok = (a.value, b.value) == (1, 128)
    print("    capi.so  로드 OK, para_range(1,256,2,0)=(%d,%d) %s"
          % (a.value, b.value, "" if ok else "<- 예상 (1,128) 과 다름!"))
else:
    print("    capi.so  아직 없음 -> 'make' 로 빌드하세요")

def show(n, f):
    try: print("    %-8s %s" % (n, f()))
    except Exception as e: print("    %-8s ERR: %s" % (n, e))
print("    python   %s" % sys.version.split()[0])
show("numpy",  lambda: __import__("numpy").__version__)
show("cupy",   lambda: __import__("cupy").__version__)
def _mpi():
    # py2f() 핸들이 통하려면 파이썬과 .so 가 같은 libmpi 를 써야 한다.
    from mpi4py import MPI
    return MPI.Get_library_version().strip().splitlines()[0]
show("mpi4py", _mpi)
PY

    echo
    echo "==> 환경 준비 완료."
    echo "    - 매 세션:  source ${here}/setup_env.sh"
    echo "    - 라이브러리 빌드: make  (lib/libpoisson_capi.so)"
    return 0
}

# 실행(exec) 시 경고 후 진행, source 시 그대로 현재 셸에 반영
if [ -n "${BASH_SOURCE:-}" ] && [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "[!] 'source 02_Python/setup_env.sh' 로 실행하세요 (module/conda 활성화가 셸에 남도록)."
    _poisson_setup; exit $?
else
    _poisson_setup
fi
