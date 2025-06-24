#!/bin/bash
set -e

cd /workspaces/GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn
sed -i 's/value.type()/value.scalar_type()/g' ms_deform_attn_cuda.cu
sed -i 's/value.scalar_type().is_cuda()/value.is_cuda()/g' ms_deform_attn_cuda.cu

cd /workspaces/GroundingDINO

# Clean any previous builds to force recompilation
rm -rf build/ **/*.so **/*.cpp **/*.cu.o
pip install -q -e .
