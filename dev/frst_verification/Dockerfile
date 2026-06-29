# Image for cross-checking the CYTools-free verifier against CYTools ground truth.
# Builds on the public CYTools image and adds CPU PyTorch + pycddlib (the deps the
# package needs that the CYTools base lacks).
#
# Build:
#   docker build -t cytools-frst dev/frst_verification
#
# Run (from the repo root; mount the repo and a folder with labeled .npy data):
#   docker run --rm \
#     -v "$PWD:/work" -v "/path/to/data:/data" \
#     -w /work -e PYTHONPATH=/work -e PYTHONUNBUFFERED=1 cytools-frst \
#     python dev/frst_verification/compare_to_cytools.py \
#       --poly /data/toSave_poly_fast_200.npy \
#       --mask /data/toSave_poly_mask_fast_200.npy \
#       --ts   /data/toSave_Ts_fast_200.npy \
#       --padding_idx 128 [--limit 60]
FROM liammcallistergroup/cytools:singularity
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir pycddlib matplotlib ijson pyyaml
