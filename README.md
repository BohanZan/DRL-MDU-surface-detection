# DRL Surface Detection — MAPPO for Multi-MDU Cooperative Coverage

Deep Reinforcement Learning (MAPPO) for cooperative path planning of multiple
**Movable Detection Units (MDUs)** traversing a space net enveloping asteroid
Bennu. Each MDU carries an 80° conical FOV sensor. Goal: maximize surface
coverage with minimal scan time.

## Architecture

```
src/
├── config.py               # Single source of truth for ALL parameters
├── run_manager.py           # Unified output path management
├── env/
│   ├── asteroid.py          # Bennu polyhedron mesh + cone FOV detection
│   ├── net_graph.py         # Space net topology (369 nodes / 432 edges)
│   └── mdu_coverage_env.py  # Gymnasium Env: reset / step / obs / reward
├── agents/
│   └── mappo.py             # MAPPO + GRU Actor + Critic + BPTT
├── train.py                 # Main training script
├── generate_trajectory.py   # Trajectory generation from checkpoint
└── visualize_trajectory.py  # Animation rendering → GIF
```

**Algorithm**: Multi-Agent PPO (MAPPO) with CTDE (Centralized Training,
Decentralized Execution), GRU temporal memory, truncated BPTT, GAE,
and a Gompertz S-curve coverage bonus.

## Demo

### Training Curves
![Training Curves](Seeings/training_curves.png)

### 4-MDU Trajectory Animation
![Animation](Seeings/Animation.gif)

## Quick Start

```bash
# Environment
conda activate comm_python_env
set KMP_DUPLICATE_LIB_OK=TRUE

# Train (4 MDUs, all defaults from src/config.py)
python src/train.py --mdus 4 --episodes 500 --save-plot --tag my_run

# Generate trajectory from checkpoint
python src/generate_trajectory.py --mode mappo --tag my_run

# Render animation
python src/visualize_trajectory.py --tag my_run
```

Output goes to `results/<timestamp>_my_run/` containing checkpoints, plots,
trajectories (NPZ + TXT), and animations (GIF).

## Key Results (v14, 500 episodes)

| Metric | Value |
|--------|-------|
| Best Coverage | 75.82% |
| Greedy Coverage | 70.02% |
| Random Baseline | ~53% |
| GRU Hidden Norm | 2.5 → 7.4 (peak) |

## Runtime Environment

Snapshot of the existing `comm_python_env` Conda environment (2026-09-12).

| Item | Version |
|------|---------|
| Operating system | Windows 11 Pro 25H2, 64-bit (OS build 26200.9445) |
| Python | 3.13.7, 64-bit |

[requirements.txt](requirements.txt) pins all 109 installed Python packages,
including the CUDA 12.8 builds of PyTorch and torchvision. This is a full snapshot,
not a minimal dependency list. The 119 Conda package records are listed separately
because pip requirements do not capture Conda's native libraries or build variants.
These records describe the existing mixed Conda/pip environment, not a tested
clean-install lockfile. The Python version above is reported by the running
interpreter; the Conda table preserves package metadata verbatim.

<details>
<summary>All Python package versions (109 packages; matches requirements.txt)</summary>

```text
aria2==0.0.1b0
asttokens==3.0.1
attrs==26.1.0
beautifulsoup4==4.14.3
Bottleneck==1.4.2
cairocffi==1.7.1
CairoSVG==2.9.0
certifi==2025.11.12
cffi==2.1.0
charset-normalizer==3.4.4
cloudpickle==3.1.2
colorama==0.4.6
comm==0.2.3
contourpy==1.3.1
cssselect2==0.9.0
cycler==0.11.0
cyclopts==4.22.2
debugpy==1.8.19
decorator==5.2.1
defusedxml==0.7.1
docstring_parser==0.18.0
executing==2.2.1
Farama-Notifications==0.0.6
filelock==3.20.0
fonttools==4.60.1
fsspec==2025.12.0
gymnasium==1.3.0
huggingface-hub==0.36.0
idna==3.11
ipykernel==7.1.0
ipython==9.8.0
ipython_pygments_lexers==1.1.1
jedi==0.19.2
Jinja2==3.1.6
joblib==1.5.3
jupyter_client==8.7.0
jupyter_core==5.9.1
kiwisolver==1.4.8
lxml==6.1.1
markdown-it-py==4.2.0
MarkupSafe==2.1.5
matplotlib==3.10.6
matplotlib-inline==0.2.1
mdurl==0.1.2
mkl_fft==2.1.1
mkl_random==1.3.0
mkl-service==2.5.2
mpmath==1.3.0
nest_asyncio==1.6.0
networkx==3.6.1
numexpr==2.14.1
numpy==2.3.4
packaging==25.0
pandas==2.3.3
parso==0.8.5
pillow==11.1.0
pip==25.2
platformdirs==4.5.1
pooch==1.9.0
prompt_toolkit==3.0.52
psutil==7.2.0
pure_eval==0.2.3
pycparser==3.0
pyDOE==0.3.8
Pygments==2.19.2
pymupdf==1.28.0
pyparsing==3.2.0
pypdf==6.0.0
PyQt6==6.9.1
PyQt6_sip==13.10.2
python-dateutil==2.9.0.post0
pytz==2025.2
pyvista==0.48.4
pywin32==311
PyYAML==6.0.2
pyzmq==27.1.0
regex==2025.11.3
reportlab==5.0.0
requests==2.32.5
rich==15.0.0
rich-rst==2.1.0
safetensors==0.7.0
scikit-learn==1.8.0
scipy==1.16.3
scooby==0.11.2
sentence-transformers==5.2.0
setuptools==78.1.1
sip==6.12.0
six==1.17.0
soupsieve==2.8.1
stack_data==0.6.3
svglib==2.0.2
sympy==1.14.0
threadpoolctl==3.6.0
tinycss2==1.5.1
tokenizers==0.22.1
torch==2.9.1+cu128
torchvision==0.24.1+cu128
tornado==6.5.1
tqdm==4.67.1
traitlets==5.14.3
transformers==4.57.3
typing_extensions==4.15.0
tzdata==2025.2
urllib3==2.6.2
vtk==9.6.2
wcwidth==0.2.14
webencodings==0.5.1
wheel==0.45.1
```

</details>

<details>
<summary>All Conda package versions and builds (119 records)</summary>

| Package | Version | Build |
|---------|---------|-------|
| _python_abi3_support | 1.0 | hd8ed1ab_2 |
| asttokens | 3.0.1 | pyhd8ed1ab_0 |
| blas | 1.0 | mkl |
| bottleneck | 1.4.2 | py313h540bb41_1 |
| bzip2 | 1.0.8 | h2bbff1b_6 |
| ca-certificates | 2025.12.2 | haa95532_0 |
| cairo | 1.18.4 | he9e932c_0 |
| colorama | 0.4.6 | pyhd8ed1ab_1 |
| comm | 0.2.3 | pyhe01879c_0 |
| contourpy | 1.3.1 | py313h214f63a_0 |
| cpython | 3.13.11 | py313hd8ed1ab_100 |
| cycler | 0.11.0 | pyhd3eb1b0_0 |
| debugpy | 1.8.19 | py313h927ade5_0 |
| decorator | 5.2.1 | pyhd8ed1ab_0 |
| executing | 2.2.1 | pyhd8ed1ab_0 |
| expat | 2.7.1 | h8ddb27b_0 |
| fontconfig | 2.15.0 | hd211d86_0 |
| fonttools | 4.60.1 | py313h02ab6af_0 |
| freetype | 2.13.3 | h0620614_0 |
| graphite2 | 1.3.14 | hd77b12b_1 |
| harfbuzz | 10.2.0 | he2f9f60_1 |
| icc_rt | 2022.1.0 | h6049295_2 |
| icu | 73.1 | h6c2663c_0 |
| intel-openmp | 2025.0.0 | haa95532_1164 |
| ipykernel | 7.1.0 | pyh6dadd2b_0 |
| ipython | 9.8.0 | pyhe2676ad_0 |
| ipython_pygments_lexers | 1.1.1 | pyhd8ed1ab_0 |
| jedi | 0.19.2 | pyhd8ed1ab_1 |
| jpeg | 9f | ha349fce_0 |
| jupyter_client | 8.7.0 | pyhcf101f3_0 |
| jupyter_core | 5.9.1 | pyh6dadd2b_0 |
| kiwisolver | 1.4.8 | py313h5da7b33_0 |
| krb5 | 1.21.3 | hdf4eb48_0 |
| lcms2 | 2.16 | hb4a4139_0 |
| lerc | 3.0 | hd77b12b_0 |
| libdeflate | 1.17 | h2bbff1b_1 |
| libffi | 3.4.4 | hd77b12b_1 |
| libglib | 2.84.4 | hfaec014_0 |
| libiconv | 1.16 | h2bbff1b_3 |
| libkrb5 | 1.21.3 | h885b0b7_4 |
| libmpdec | 4.0.0 | h827c3e9_0 |
| libpng | 1.6.50 | h46444df_0 |
| libpq | 17.6 | h652a1e2_0 |
| libsodium | 1.0.20 | hc70643c_0 |
| libtiff | 4.5.1 | hd77b12b_0 |
| libwebp-base | 1.3.2 | h3d04722_1 |
| libxml2 | 2.13.9 | h6201b9f_0 |
| libzlib | 1.3.1 | h02ab6af_0 |
| lz4-c | 1.9.4 | h2bbff1b_1 |
| matplotlib | 3.10.6 | py313haa95532_1 |
| matplotlib-base | 3.10.6 | py313h26e45b9_1 |
| matplotlib-inline | 0.2.1 | pyhd8ed1ab_0 |
| mkl | 2025.0.0 | h5da7b33_930 |
| mkl_fft | 2.1.1 | py313hbc2a22c_0 |
| mkl_random | 1.3.0 | py313h42c1672_0 |
| mkl-service | 2.5.2 | py313h0b37514_0 |
| mysql-common | 9.3.0 | hf582a5b_3 |
| mysql-libs | 9.3.0 | hc0ebf12_3 |
| nest-asyncio | 1.6.0 | pyhd8ed1ab_1 |
| numexpr | 2.14.1 | py313h7660c64_0 |
| numpy | 2.3.4 | py313h050da96_1 |
| numpy-base | 2.3.4 | py313h1e017a8_1 |
| openjpeg | 2.5.2 | hae555c5_0 |
| openssl | 3.6.0 | h725018a_0 |
| packaging | 25.0 | py313haa95532_1 |
| pandas | 2.3.3 | py313h42c1672_1 |
| parso | 0.8.5 | pyhcf101f3_0 |
| pcre2 | 10.46 | h5740b90_0 |
| pillow | 11.1.0 | py313h096bfcc_0 |
| pip | 25.2 | pyhc872135_0 |
| pixman | 0.46.4 | h4043f72_0 |
| platformdirs | 4.5.1 | pyhcf101f3_0 |
| prompt-toolkit | 3.0.52 | pyha770c72_0 |
| psutil | 7.2.0 | py313h5fd188c_0 |
| pure_eval | 0.2.3 | pyhd8ed1ab_1 |
| pygments | 2.19.2 | pyhd8ed1ab_0 |
| pyparsing | 3.2.0 | py313haa95532_0 |
| pypdf | 6.0.0 | py313haa95532_0 |
| pyqt | 6.9.1 | py313h12ec796_0 |
| pyqt6-sip | 13.10.2 | py313h630b2a1_0 |
| python | 3.13.7 | h260b955_100_cp313 |
| python_abi | 3.13 | 1_cp313 |
| python-dateutil | 2.9.0post0 | py313haa95532_2 |
| python-gil | 3.13.11 | h4df99d1_100 |
| python-tzdata | 2025.2 | pyhd3eb1b0_0 |
| pytz | 2025.2 | py313haa95532_0 |
| pywin32 | 311 | py313h40c08fc_1 |
| pyyaml | 6.0.2 | py313h827c3e9_0 |
| pyzmq | 27.1.0 | py312hbb5da91_0 |
| qtbase | 6.9.2 | h06bae2a_4 |
| qtdeclarative | 6.9.2 | h88b4c33_1 |
| qtsvg | 6.9.2 | h30ace32_1 |
| qttools | 6.9.2 | h7e7b719_1 |
| qtwebchannel | 6.9.2 | heb02b0b_1 |
| qtwebsockets | 6.9.2 | heb02b0b_1 |
| scipy | 1.16.3 | py313hbd6d557_0 |
| setuptools | 78.1.1 | py313haa95532_0 |
| sip | 6.12.0 | py313h706e071_0 |
| six | 1.17.0 | py313haa95532_0 |
| sqlite | 3.50.2 | hda9a48d_1 |
| stack_data | 0.6.3 | pyhd8ed1ab_1 |
| tbb | 2022.0.0 | h214f63a_0 |
| tbb-devel | 2022.0.0 | h214f63a_0 |
| tk | 8.6.15 | hf199647_0 |
| tornado | 6.5.1 | py313h827c3e9_0 |
| traitlets | 5.14.3 | pyhd8ed1ab_1 |
| typing_extensions | 4.15.0 | pyhcf101f3_0 |
| tzdata | 2025b | h04d1e81_0 |
| ucrt | 10.0.22621.0 | haa95532_0 |
| vc | 14.3 | h2df5915_10 |
| vc14_runtime | 14.44.35208 | h4927774_10 |
| vs2015_runtime | 14.44.35208 | ha6b5a95_10 |
| wcwidth | 0.2.14 | pyhd8ed1ab_0 |
| wheel | 0.45.1 | py313haa95532_0 |
| xz | 5.6.4 | h4754444_1 |
| yaml | 0.2.5 | he774522_0 |
| zeromq | 4.3.5 | h5bddc39_9 |
| zlib | 1.3.1 | h02ab6af_0 |
| zstd | 1.5.7 | h56299aa_0 |

</details>

## Data Files

- `FNS_square_fold-50m.txt` — space net topology
- `Solution.dat` — final net state after asteroid capture
- `polyhedron_bennu.txt` — Bennu mesh (1348 vertices, 2692 faces)

## License

MIT
