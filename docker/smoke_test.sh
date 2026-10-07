#!/usr/bin/env bash
# Host + small-shape ASC/PTO smoke tests. NPU cases run under cannsim.
set -euo pipefail

CANN_SET_ENV="$(ls -d /usr/local/Ascend/cann-*/set_env.sh 2>/dev/null | sort | tail -1 || true)"
if [[ -n "${CANN_SET_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${CANN_SET_ENV}"
fi
export LD_LIBRARY_PATH="/usr/local/Ascend/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver_libs:${LD_LIBRARY_PATH:-}"

echo "== import check =="
python -c "import tilelang, torch, torch_npu; print('tilelang', tilelang.__version__); print('torch', torch.__version__)"
command -v ptoas
if ! command -v cannsim >/dev/null 2>&1; then
  echo "cannsim not found in PATH; NPU smoke tests require the CANN simulator." >&2
  exit 1
fi

TESTING_DIR=/opt/tilelang/testing
PYTEST_COMMON=(python -m pytest -q --tb=short --import-mode=importlib --rootdir="${TESTING_DIR}")

echo "== host-only tests (CPU / compile-only) =="
"${PYTEST_COMMON[@]}" \
  "${TESTING_DIR}/python/cpu/test_tilelang_cpu_fp16_const.py" \
  "${TESTING_DIR}/python/cpu/test_tilelang_cpu_reduce.py::test_cpu_reduce_2d_float" \
  "${TESTING_DIR}/ascend/target/test_tilelang_ascend_target.py" \
  "${TESTING_DIR}/ascend/language/test_tilelang_ascend_copy.py" \
  "${TESTING_DIR}/ascend/language/test_tilelang_ascend_pto_types.py::test_pto_float32x2_minmax_codegen"

# cannsim/npusim launches python with cwd set to the interpreter bindir, so
# pytest paths must be absolute. Keep device inputs small to make CPU simulation
# practical.
echo "== NPU tests via cannsim (Ascend950, no performance report) =="
set +e
cannsim_log="$(mktemp)"
cannsim record "$(command -v python)" -s Ascend950 -u "-m pytest -q --tb=short --import-mode=importlib --rootdir=${TESTING_DIR} ${TESTING_DIR}/ascend/language/test_tilelang_ascend_warp_vote.py ${TESTING_DIR}/ascend/language/test_tilelang_ascend_pto_intrinsics.py::test_pto_pairwise_sum ${TESTING_DIR}/ascend/language/test_tilelang_ascend_pto_intrinsics.py::test_pto_cast_roundtrip" 2>&1 | tee "${cannsim_log}"
cannsim_status=${PIPESTATUS[0]}
set -e
if grep -qE '(^| )[1-9][0-9]* failed' "${cannsim_log}"; then
  echo "NPU pytest reported failures (cannsim exit=${cannsim_status})" >&2
  rm -f "${cannsim_log}"
  exit 1
fi
if ! grep -qE '[1-9][0-9]* passed' "${cannsim_log}"; then
  echo "NPU pytest did not report a passing summary (cannsim exit=${cannsim_status})" >&2
  rm -f "${cannsim_log}"
  exit 1
fi
if [[ "${cannsim_status}" -ne 0 ]]; then
  echo "NPU pytest passed; ignoring cannsim teardown status ${cannsim_status} (known SIGSEGV after user app exits)"
fi
rm -f "${cannsim_log}"

echo "== smoke tests passed =="
