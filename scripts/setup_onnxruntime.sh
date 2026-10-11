#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly DEPS_ROOT="${VISION_ENCODERS_DEPS_DIR:-${REPO_ROOT}/.deps}"
readonly CACHE_ROOT="${VISION_ENCODERS_CACHE_DIR:-${REPO_ROOT}/.cache}"
readonly LINK_PATH="${REPO_ROOT}/thirdparty/onnxruntime"

arch_override=""
cuda_major_override=""
archive_path=""
archive_sha256=""
dry_run=0
force=0
build_package=0

usage() {
  cat <<'EOF'
Usage: scripts/setup_onnxruntime.sh [options]

Install a pinned GPU-enabled ONNX Runtime and create thirdparty/onnxruntime.

Options:
  --archive PATH       Install a local prebuilt archive instead of downloading/building
  --sha256 HEX         Expected checksum for --archive
  --build              Build the vision_encoders ROS package after setup
  --force              Replace an existing managed installation (keeps a backup)
  --dry-run            Print the selected installation without changing anything
  --arch ARCH          Override architecture for dry-run testing (x86_64 or aarch64)
  --cuda-major N       Override detected CUDA major for dry-run/source selection
  -h, --help           Show this help

x86_64 downloads Microsoft's pinned GPU archive. Linux aarch64 GPU archives are
not published by Microsoft, so Jetson installs build a pinned release from source.
Use --archive for an internally validated Jetson package when one is available.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

note() {
  echo "setup_onnxruntime: $*"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

normalize_arch() {
  case "$1" in
    x86_64|amd64) echo "x86_64" ;;
    aarch64|arm64) echo "aarch64" ;;
    *) die "unsupported architecture: $1" ;;
  esac
}

while (($# > 0)); do
  case "$1" in
    --archive)
      (($# >= 2)) || die "--archive requires a path"
      archive_path="$2"
      shift 2
      ;;
    --sha256)
      (($# >= 2)) || die "--sha256 requires a value"
      archive_sha256="$2"
      shift 2
      ;;
    --build) build_package=1; shift ;;
    --force) force=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --arch)
      (($# >= 2)) || die "--arch requires a value"
      arch_override="$2"
      shift 2
      ;;
    --cuda-major)
      (($# >= 2)) || die "--cuda-major requires a value"
      cuda_major_override="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

host_arch="$(normalize_arch "$(uname -m)")"
selected_arch="$(normalize_arch "${arch_override:-${host_arch}}")"
if [[ "${selected_arch}" != "${host_arch}" && ${dry_run} -eq 0 ]]; then
  die "--arch may differ from the host only with --dry-run"
fi

ort_version=""
ort_commit=""
package_name=""
download_url=""
download_sha256=""
install_mode=""
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
cudnn_home="${CUDNN_HOME:-/usr}"

detect_cuda_major() {
  if [[ -n "${cuda_major_override}" ]]; then
    [[ "${cuda_major_override}" =~ ^[0-9]+$ ]] || die "invalid CUDA major: ${cuda_major_override}"
    echo "${cuda_major_override}"
    return
  fi
  [[ -x "${cuda_home}/bin/nvcc" ]] || die \
    "nvcc not found under ${cuda_home}; set CUDA_HOME or pass --cuda-major for --dry-run"
  local detected
  detected="$("${cuda_home}/bin/nvcc" --version |
    sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | tail -n 1)"
  [[ -n "${detected}" ]] || die "unable to detect CUDA version from ${cuda_home}/bin/nvcc"
  echo "${detected}"
}

if [[ "${selected_arch}" == "x86_64" ]]; then
  ort_version="1.22.0"
  ort_commit="f217402897f40ebba457e2421bc0a4702771968e"
  package_name="onnxruntime-linux-x64-gpu-${ort_version}"
  download_url="https://github.com/microsoft/onnxruntime/releases/download/v${ort_version}/${package_name}.tgz"
  download_sha256="2a19dbfa403672ec27378c3d40a68f793ac7a6327712cd0e8240a86be2b10c55"
  install_mode="prebuilt"
else
  cuda_major="$(detect_cuda_major)"
  if ((cuda_major >= 12)); then
    ort_version="1.22.0"
    ort_commit="f217402897f40ebba457e2421bc0a4702771968e"
  elif ((cuda_major == 11)); then
    ort_version="1.15.1"
    ort_commit="baeece44ba075009c6bfe95891a8c1b3d4571cb3"
  else
    die "CUDA ${cuda_major} is unsupported; expected CUDA 11 or newer"
  fi
  package_name="onnxruntime-linux-aarch64-gpu-${ort_version}"
  install_mode="source"
fi

install_root="${DEPS_ROOT}/onnxruntime/${package_name}"

if [[ -n "${archive_path}" ]]; then
  install_mode="archive"
  if [[ "${selected_arch}" == "aarch64" && -z "${archive_sha256}" ]]; then
    die "a custom aarch64 archive requires --sha256"
  fi
fi

note "architecture: ${selected_arch}"
note "ONNX Runtime: ${ort_version} (${install_mode})"
note "install root: ${install_root}"
note "stable link: ${LINK_PATH}"
if [[ "${install_mode}" == "prebuilt" ]]; then
  note "source: ${download_url}"
elif [[ "${install_mode}" == "source" ]]; then
  note "source commit: ${ort_commit}"
  note "CUDA: ${cuda_home}; cuDNN: ${cudnn_home}"
else
  note "archive: ${archive_path}"
fi
if ((dry_run)); then
  exit 0
fi

validate_install() {
  local root="$1"
  [[ -f "${root}/include/onnxruntime_cxx_api.h" ]] || return 1
  [[ -f "${root}/lib/libonnxruntime.so" ]] || return 1
  [[ -f "${root}/lib/libonnxruntime_providers_cuda.so" ]] || return 1
}

prepare_install_root() {
  if [[ ! -e "${install_root}" ]]; then
    return
  fi
  if validate_install "${install_root}" && ((force == 0)); then
    note "reusing existing validated installation"
    return 2
  fi
  ((force)) || die "${install_root} exists but is incomplete; rerun with --force"
  local backup="${install_root}.backup.$(date +%Y%m%d%H%M%S)"
  mv -- "${install_root}" "${backup}"
  note "moved previous installation to ${backup}"
}

verify_archive() {
  local path="$1"
  local expected="$2"
  [[ -f "${path}" ]] || die "archive not found: ${path}"
  if [[ -n "${expected}" ]]; then
    require_command sha256sum
    local actual
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]] || die \
      "checksum mismatch for ${path}: expected ${expected}, got ${actual}"
  fi
}

install_archive() {
  local path="$1"
  local expected_sha="$2"
  verify_archive "${path}" "${expected_sha}"
  mkdir -p "${DEPS_ROOT}/onnxruntime"
  local extract_root
  extract_root="$(mktemp -d "${DEPS_ROOT}/onnxruntime/.extract.XXXXXX")"
  tar -xzf "${path}" -C "${extract_root}"

  local extracted="${extract_root}/${package_name}"
  if [[ ! -d "${extracted}" ]]; then
    local -a roots=()
    mapfile -t roots < <(find "${extract_root}" -mindepth 1 -maxdepth 1 -type d)
    ((${#roots[@]} == 1)) || die "archive must contain exactly one root directory"
    extracted="${roots[0]}"
  fi
  validate_install "${extracted}" || die "archive does not contain a CUDA-enabled ONNX Runtime SDK"
  mv -- "${extracted}" "${install_root}"
  rm -rf -- "${extract_root}"
}

download_prebuilt() {
  require_command curl
  require_command tar
  require_command sha256sum
  mkdir -p "${CACHE_ROOT}"
  local archive="${CACHE_ROOT}/${package_name}.tgz"
  if [[ -f "${archive}" ]]; then
    verify_archive "${archive}" "${download_sha256}"
  else
    local partial="${archive}.partial"
    curl -fL --retry 3 --output "${partial}" "${download_url}"
    verify_archive "${partial}" "${download_sha256}"
    mv -- "${partial}" "${archive}"
  fi
  install_archive "${archive}" "${download_sha256}"
}

build_from_source() {
  require_command git
  require_command cmake
  require_command python3
  [[ -x "${cuda_home}/bin/nvcc" ]] || die "nvcc not found under ${cuda_home}"

  local source_root="${DEPS_ROOT}/src/onnxruntime-${ort_version}"
  local build_root="${DEPS_ROOT}/build/onnxruntime-${ort_version}"
  mkdir -p "${DEPS_ROOT}/src" "${DEPS_ROOT}/build" "${DEPS_ROOT}/onnxruntime"
  if [[ ! -d "${source_root}/.git" ]]; then
    git clone --branch "v${ort_version}" --depth 1 --recursive \
      https://github.com/microsoft/onnxruntime.git "${source_root}"
  fi
  local actual_commit
  actual_commit="$(git -C "${source_root}" rev-parse HEAD)"
  [[ "${actual_commit}" == "${ort_commit}" ]] || die \
    "source checkout is ${actual_commit}, expected ${ort_commit}"
  git -C "${source_root}" submodule update --init --recursive --depth 1

  "${source_root}/build.sh" \
    --config Release \
    --build_dir "${build_root}" \
    --parallel \
    --skip_tests \
    --build_shared_lib \
    --use_cuda \
    --cuda_home "${cuda_home}" \
    --cudnn_home "${cudnn_home}" \
    --cmake_extra_defines "CMAKE_INSTALL_PREFIX=${install_root}"
  cmake --install "${build_root}/Release" --prefix "${install_root}"
  validate_install "${install_root}" || die "source build completed without an installable SDK"
}

reuse_install=0
if prepare_install_root; then
  reuse_install=0
else
  status=$?
  if ((status == 2)); then
    reuse_install=1
  else
    exit "${status}"
  fi
fi

if ((reuse_install == 0)); then
  case "${install_mode}" in
    prebuilt) download_prebuilt ;;
    source) build_from_source ;;
    archive)
      expected="${archive_sha256}"
      if [[ -z "${expected}" && "${selected_arch}" == "x86_64" ]]; then
        expected="${download_sha256}"
      fi
      install_archive "${archive_path}" "${expected}"
      ;;
  esac
fi

mkdir -p "$(dirname -- "${LINK_PATH}")"
if [[ -e "${LINK_PATH}" && ! -L "${LINK_PATH}" ]]; then
  die "refusing to replace non-symlink ${LINK_PATH}"
fi
ln -sfn "../.deps/onnxruntime/${package_name}" "${LINK_PATH}"

missing_dependencies="$(
  ldd "${install_root}/lib/libonnxruntime_providers_cuda.so" 2>/dev/null |
    awk '/not found/ {print $1}' | paste -sd, -
)"
if [[ -n "${missing_dependencies}" ]]; then
  echo "warning: CUDA provider dependencies not found: ${missing_dependencies}" >&2
  echo "         install a matching CUDA/cuDNN runtime before running the encoder" >&2
fi

note "ready: ${LINK_PATH} -> $(readlink "${LINK_PATH}")"

if ((build_package)); then
  require_command colcon
  ros_distro="${ROS_DISTRO:-humble}"
  ros_setup="/opt/ros/${ros_distro}/setup.bash"
  [[ -f "${ros_setup}" ]] || die "ROS setup not found: ${ros_setup}"
  # shellcheck disable=SC1090
  set +u
  source "${ros_setup}"
  set -u
  if [[ "$(basename -- "$(dirname -- "${REPO_ROOT}")")" == "src" ]]; then
    workspace_root="$(cd -- "${REPO_ROOT}/../.." && pwd)"
    (cd "${workspace_root}" && colcon build --symlink-install \
      --packages-select vision_encoders \
      --cmake-args "-DONNXRUNTIME_ROOT=${install_root}")
  else
    colcon --log-base "${REPO_ROOT}/.colcon/log" build \
      --base-paths "${REPO_ROOT}" \
      --build-base "${REPO_ROOT}/.colcon/build" \
      --install-base "${REPO_ROOT}/.colcon/install" \
      --symlink-install \
      --packages-select vision_encoders \
      --cmake-args "-DONNXRUNTIME_ROOT=${install_root}"
    note "source ${REPO_ROOT}/.colcon/install/setup.bash"
  fi
fi
