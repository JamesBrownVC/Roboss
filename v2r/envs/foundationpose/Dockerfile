# syntax=docker/dockerfile:1
# V2R stage env: objects 6DoF pose (FoundationPose).
# Docker (not pixi) because the native build is fragile: custom CUDA
# extensions, nvdiffrast, kaolin, eigen/boost pybind modules.
#
# ============================ LICENSE WARNING ============================
# FoundationPose is released under the NVIDIA non-commercial source code
# license (config/licensing.yaml: commercial: no). BundleSDF (mesh-free
# companion) is likewise non-commercial. When licensing.permissive_only=true
# the pipeline DOES NOT run this image at all; it falls back to
# mask + depth ICP tracking (Open3D, custom) with conf_multiplier 0.6.
# =========================================================================

FROM nvcr.io/nvidia/pytorch:23.10-py3

# TODO(operator): pin to the exact commit; recorded in the objects manifest.
ARG PINNED_COMMIT=PINNED_COMMIT_FOUNDATIONPOSE
ARG FOUNDATIONPOSE_REPO=https://github.com/NVlabs/FoundationPose.git

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build \
        libeigen3-dev libboost-all-dev libgl1 libglib2.0-0 \
        libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone ${FOUNDATIONPOSE_REPO} /opt/foundationpose \
    && git -C /opt/foundationpose checkout ${PINNED_COMMIT}

WORKDIR /opt/foundationpose

# Python deps per the repo README at the pinned commit (README wins on
# installation details). nvdiffrast + kaolin are the usual build hazards.
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir \
        "git+https://github.com/NVlabs/nvdiffrast.git"

# Build the bundled C++/CUDA extensions (mycpp / mycuda). This is the fragile
# step that justifies Docker; do not attempt it in a shared env.
RUN bash build_all.sh

# ---------------------------------------------------------------------------
# Weights: FoundationPose model weights and demo data are distributed under
# the NVIDIA non-commercial license. The OPERATOR must review and accept the
# license, then download into ./weights (mounted or baked in a derived
# image). COMMENTED OUT on purpose -- never fetched automatically:
#
# RUN python tools/download_weights.py --out /opt/foundationpose/weights
#
# At runtime the stage wrapper mounts:
#   -v {workspace}:/data          (episode workspace, contract IO)
#   -v {weights_dir}:/opt/foundationpose/weights:ro
# ---------------------------------------------------------------------------

ENTRYPOINT ["python"]
