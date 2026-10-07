# https://github.com/Ascend/cann-container-image/tree/main/cann/9.2.0-beta.2-950-ubuntu22.04-py3.12
FROM quay.io/ascend/cann:9.2.0-beta.2-950-ubuntu22.04-py3.12

RUN apt-get update && apt-get install -y \
    wget curl git vim gawk ripgrep \
    ninja-build cmake ccache build-essential lld zip binutils patchelf chrpath \
    && rm -rf /var/lib/apt/lists/*

# NOTE: need to further update torch_npu for CANN 9.2.0? https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md
RUN pip install --no-cache-dir torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir torch-npu==2.9.0.post6 \
    && pip install --no-cache-dir numpy pyyaml

# install tilelang here
# LLVM/MLIR Python bindings are not yet compatible with pybind11 3.x
RUN pip install --no-cache-dir pytest 'pybind11==2.12.0'

# Self-contained PTO toolchain from hw-native-sys/PTOAS vmi-v0.1.9 (cp312 matches this image).
RUN set -eux; \
    arch="$(uname -m)"; \
    case "${arch}" in \
      x86_64) \
        PTOAS_ARCH=x86_64; \
        PTOAS_SHA256=475b981525f5a1129f7ab89a63670da8375bd00a4e4764dd796b7bea02ac0a35; \
        ;; \
      aarch64) \
        PTOAS_ARCH=aarch64; \
        PTOAS_SHA256=db8e3c5419de939ed02a68bbc52756862b21fa4f607d9395a399d05862593c34; \
        ;; \
      *) \
        echo "Unsupported architecture for ptoas: ${arch}" >&2; \
        exit 1; \
        ;; \
    esac; \
    PTOAS_FILENAME="ptoas_vmi-0.1.9-cp312-cp312-manylinux_2_34_${PTOAS_ARCH}.whl"; \
    curl --fail --location --retry 3 \
      --output "/tmp/${PTOAS_FILENAME}" \
      "https://github.com/hw-native-sys/PTOAS/releases/download/vmi-v0.1.9/${PTOAS_FILENAME}"; \
    printf '%s  %s\n' "${PTOAS_SHA256}" "/tmp/${PTOAS_FILENAME}" | sha256sum --check -; \
    pip install --no-cache-dir "/tmp/${PTOAS_FILENAME}"; \
    rm -f "/tmp/${PTOAS_FILENAME}"

COPY third_party/tilelang /opt/tilelang
ENV USE_CUDA=0 \
    USE_ROCM=0 \
    CMAKE_GENERATOR=Ninja
RUN cd /opt/tilelang \
    && CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" pip install --no-cache-dir -v .

# ==== github cli https://github.com/cli/cli/blob/trunk/docs/install_linux.md

RUN mkdir -p -m 755 /etc/apt/keyrings \
	&& out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
	&& cat $out | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& mkdir -p -m 755 /etc/apt/sources.list.d \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" |  tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& apt update \
	&& apt install gh -y

# ==== AI agent CLIs (install last for better layer caching) ====

# cursor CLI
RUN curl -fsSL https://cursor.com/install | bash \
    && ln -sf /root/.local/bin/agent /usr/local/bin/agent

# Codex CLI
RUN CODEX_NON_INTERACTIVE=1 curl -fsSL https://chatgpt.com/codex/install.sh | bash \
    && ln -sf /root/.local/bin/codex /usr/local/bin/codex

# Claude Code CLI
RUN curl -fsSL https://claude.ai/install.sh | bash \
    && ln -sf /root/.local/bin/claude /usr/local/bin/claude

COPY docker/smoke_test.sh /usr/local/bin/tilelang-smoke
RUN chmod +x /usr/local/bin/tilelang-smoke
