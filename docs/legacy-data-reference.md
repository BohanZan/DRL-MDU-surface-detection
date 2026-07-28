# Legacy Code Data Structures Reference

> Reference for the FNS_capture codebase (IAA 2025). These data structures define
> the input for our MDU path planning project — the final net state becomes our
> starting point.

---

## 1. FNS Topology File — The Net Structure

**Location:** `C:\Users\Lenovo\Desktop\Study\AsteroidResearch\FNS_capture_control\FNS_capture\FNS_capture\FNS\`
**Source:** `LoadFNS.h` / `LoadFNS.cpp`

### FNS Struct (C++ definition)

```cpp
typedef struct FlexibleNetSpacecraft {
    int NumPoints;      // Number of nodes in the net mesh
    int NumEdges;       // Number of edges (cables connecting nodes)
    int NumRopeNodes;   // Number of "rope" boundary nodes
    int col_Topo;       // Columns in the topology (adjacency) matrix
    double Radius;      // Characteristic radius of the net
    int *Ed;            // Edge connectivity [2 × NumEdges]
    int *Topo;          // Topology matrix [NumPoints × col_Topo]
    double *Pt;         // Node positions [3 × NumPoints] = (x, y, z)
    double *EL;         // Edge rest lengths [NumEdges]
    double *BL;         // Boundary length matrix [NumPoints × NumPoints]
} FNS;
```

### File Layout (ASCII, space-separated)

```
Line 1:      NumPoints NumEdges NumRopeNodes col_Topo Radius
Next blocks:
  [Edges]    NumEdges lines, each: node_i node_j  (0-indexed)
  [Topo]     NumPoints rows × col_Topo columns — adjacency/connectivity
  [Pt]       NumPoints rows, each: x y z
  [EL]       NumEdges values (one per line or space-separated)
  [BL]       NumPoints × NumPoints matrix (row-major)
```

### Available FNS Files

| File | NumPoints | NumEdges | NumRopeNodes | Radius (m) | Description |
|------|-----------|----------|-------------|------------|-------------|
| `FNS_square.txt` | 369 | 432 | 288 | 492.20 | Standard square net (flat, full-size) |
| `FNS_square_fold-80m.txt` | 369 | 432 | 288 | 492.20 | Folded square, 80m deployment |
| `FNS_square_fold-50m.txt` | 369 | 432 | 288 | 492.20 | Folded square, 50m deployment |
| `FNS_square_fold-200m.txt` | 369 | 432 | 288 | 492.20 | Folded square, 200m deployment |
| `FNS_spider.txt` | 369 | 432 | 288 | — | Spider-web topology |
| `FNS_ThreeDetector.txt` | 129 | 172 | 100 | — | 3-detector variant |
| `FNS_particle_earth.txt` | 3 | 3 | 0 | 62.28 | Tiny test case (3 nodes) |

### Actuator Node Indices

The 4 actuators (control points for capture) are at node indices:
```cpp
const int index_of_actuator[4] = { 288, 296, 360, 368 };
```

These are the "corners" of the net. In the new project, they may serve as
MDU deployment points or reference anchors.

### Edge Data Layout (Ed array)

`Ed[0..NumEdges-1]` = first endpoint of each edge
`Ed[NumEdges..2*NumEdges-1]` = second endpoint of each edge

So edge `i` connects node `Ed[i]` ↔ `Ed[NumEdges + i]`.

### Topo Matrix Layout

`Topo[i + j * NumPoints]` where `i` = node index, `j` = neighbor slot (0..col_Topo-1).
Value `-1` means "no neighbor" in that slot.
Positive values are connected node indices.

---

## 2. Solution Files — Final Net State (Project Starting Point)

**Location:** `C:\Users\Lenovo\Desktop\Study\AsteroidResearch\FNS_capture_control\FNS_capture\FNS_capture\Solution\`
**Written by:** `OdeEuler.cpp` (line ~446)

### Format

Each row is a time-step snapshot. Output every 100 integration steps.

One row contains `6 × NumPoints` values:
```
[ positions (3×NumPoints) | velocities (3×NumPoints) ]
```

Where positions = `x0 y0 z0  x1 y1 z1  ...  x{N-1} y{N-1} z{N-1}`
And velocities = `vx0 vy0 vz0  vx1 vy1 vz1  ...  vx{N-1} vy{N-1} vz{N-1}`

**The last row is the final state** — this is what we use as the fixed net
geometry for MDU path planning.

### Key Solution Files (Bennu)

| File | Description |
|------|-------------|
| `last.txt` | Final converged state (directly usable as starting point) |
| `Bennu_fold-long.txt` | Bennu, long deployment, folded net |
| `Bennu_fold-short.txt` | Bennu, short deployment |
| `Bennu_fold-slippage-*.txt` | Slippage scenarios |
| `validation_square_Didymos.txt` | Validation: Didymos (in Solution/) |
| `validation_square_gelovka.txt` | Validation: Gelovka (in Solution/) |
| `validation_square_Bennu.txt` | Validation: Bennu (in MATLAB directory) |

### Key Solution Files (Didymos)

| File | Description |
|------|-------------|
| `Didymos-*-1E-3.txt` | Various Didymos scenarios |
| `Didymos_fold-long.txt` | Didymos folded long |

### ⚠️ Coordinate System Warning

From examining actual solution files (`Bennu_fold-long.txt`, `last.txt`):
- Net node positions are **NOT** in the asteroid body frame (centered at origin)
- Net positions have y ≈ +1300, meaning the simulation frame is translated relative
  to the asteroid frame (which is centered at ~0,0,0 with radius ~250-300m)
- **For MDU path planning**: either subtract the offset to align with asteroid center,
  or work in the simulation frame and compute relative vectors
- The asteroid SH surface is defined in body frame (origin at center of mass)
- Verify coordinate alignment when computing FOV intersections!

### Reading in Python

```python
import numpy as np

# Load final state (last row) from solution file
data = np.loadtxt("Solution/last.txt")  # shape: (n_timesteps, 6*NumPoints)
final_state = data[-1, :]               # last row

# Split into positions and velocities
n = 369  # NumPoints
positions = final_state[:3*n].reshape(n, 3)   # (n, 3) array of (x, y, z)
velocities = final_state[3*n:6*n].reshape(n, 3)

# NOTE: positions are in simulation frame (y ~ +1300 offset from asteroid center)
# To align with asteroid body frame (origin at center):
# positions_body = positions - np.mean(positions, axis=0)  # approximate
# Or use known offset: positions - [0, ~1300, 0]

# Or load initial state (first row) if needed
initial_state = data[0, :]
```

---

## 3. Asteroid Polyhedron Model

**Location:** `C:\Users\Lenovo\Desktop\Study\AsteroidResearch\FNS_capture_control\FNS_capture\FNS_capture\PolyModel\`
**Source:** `LoadPolyhedron.h` / `LoadPolyhedron.cpp`

### POLYHEDRON Struct

```cpp
typedef struct polyhedron {
    int NumVerts, NumFaces, NumEdges;
    double Density;
    int *Faces;            // [3 × NumFaces] — vertex indices per triangle
    int *Edges;            // [4 × NumEdges] — v1, v2, f1, f2
    double *Vertices;      // [3 × NumVerts] — (x, y, z)
    double *EdgeLens;      // [NumEdges]
    double *EdgeNormVecs;  // [8 × NumEdges] — edge normal vectors
    double *FaceNormVecs;  // [3 × NumFaces] — face normal vectors
} POLYHEDRON;
```

### File Layout

```
Line 1:       NumVerts NumFaces NumEdges Density
[Vertices]    NumVerts lines, each: x y z
[Faces]       NumFaces lines, each: v1 v2 v3  (vertex indices, 0-based)
[Edges]       NumEdges lines, each: v1 v2 f1 f2
[EdgeLens]    NumEdges values
[EdgeNormVecs] NumEdges × 8 values (edge normal vectors)
[FaceNormVecs] NumFaces × 3 values (face normal vectors)
```

### Available Asteroid Models

| File | NumVerts | NumFaces | NumEdges | Density | Description |
|------|----------|----------|----------|---------|-------------|
| `polyhedron_bennu.txt` | 1348 | 2692 | 4038 | 1260.0 | Bennu (primary target) |
| `polyhedron_Didymos.txt` | — | — | — | — | Didymos binary asteroid |
| `polyhedron_gelovka.txt` | — | — | — | — | Gelovka |

### Python Usage

```python
import numpy as np

with open("PolyModel/polyhedron_bennu.txt", "r") as f:
    header = f.readline().split()
    n_verts, n_faces, n_edges, density = map(float, header)
    n_verts, n_faces, n_edges = int(n_verts), int(n_faces), int(n_edges)

    verts = np.loadtxt(f, max_rows=n_verts)          # (n_verts, 3)
    faces = np.loadtxt(f, max_rows=n_faces, dtype=int) # (n_faces, 3)
    edges_data = np.loadtxt(f, max_rows=n_edges, dtype=int)  # (n_edges, 4)
    edge_lens = np.loadtxt(f, max_rows=n_edges)       # (n_edges,)
    # ... skip norm vectors for now
```

---

## 4. Spherical Harmonics Surface Parameters

**Location:** `C:\Users\Lenovo\Desktop\Study\AsteroidResearch\FNS_capture_control\FNS_capture\FNS_capture\SHPara\`
**Source:** `SHPara.h` / `LoadSHPara.cpp`

### File Format

Each line: `l  m  Clm  Slm`

Where:
- `l` = degree (0..SHlmax, where SHlmax = 25)
- `m` = order (0..l)
- `Clm` = cosine coefficient
- `Slm` = sine coefficient

### Available SH Parameter Files

| File | Description |
|------|-------------|
| `SHPara_bennu.txt` | Bennu spherical harmonics fit |
| `SHPara_Didymos.txt` | Didymos spherical harmonics fit |
| `SHPara_gelovka.txt` | Gelovka spherical harmonics fit |
| `SHPara_Mithra.txt` | Mithra (newer addition) |

### Surface Evaluation (from SurfacePoints.cpp)

The smooth asteroid surface at spherical coordinates `(θ, φ)` is:

```
R(θ, φ) = Σ_{l=0}^{SHdegree} Σ_{m=0}^{l} K_lm · P_lm(cos θ) · (C_lm · cos(mφ) + S_lm · sin(mφ))
```

Where:
- `P_lm` = associated Legendre polynomial
- `K_lm` = normalization factor: `√((2l+1)/(4π) · (l-m)!/(l+m)!)`
- `SHdegree = 21` (effective fitting degree)
- `SHlmax = 25` (storage allocation limit)

The function also returns:
- Surface normal vector `n̂` (outward-pointing)
- Two tangent vectors for local navigation

### Python Surface Evaluation

```python
import math
import numpy as np
from scipy.special import lpmv

def sh_surface(theta, phi, coeffs, degree=21):
    """Evaluate asteroid surface radius at (theta, phi).
    
    Args:
        theta: polar angle [0, pi]
        phi: azimuthal angle [0, 2*pi)
        coeffs: (N, 4) array with columns [l, m, C, S] or similar
    
    Returns:
        R: radius from asteroid center
    """
    R = 0.0
    for l, m, C, S in coeffs:
        l, m = int(l), int(m)
        if l > degree:
            continue
        # Associated Legendre polynomial
        P = lpmv(m, l, np.cos(theta))
        # Normalization
        K = math.sqrt((2*l + 1) / (4*np.pi) * math.factorial(l-m) / math.factorial(l+m))
        if m == 0:
            R += K * P * C
        else:
            R += math.sqrt(2) * K * P * (C * np.cos(m*phi) + S * np.sin(m*phi))
    return R

# Convert to Cartesian
x = R * np.sin(theta) * np.cos(phi)
y = R * np.sin(theta) * np.sin(phi)
z = R * np.cos(theta)
```

---

## 5. Physical Constants

**Location:** `C:\Users\Lenovo\Desktop\Study\AsteroidResearch\FNS_capture_control\FNS_capture\FNS_capture\constant.h`

| Constant | Value | Description |
|----------|-------|-------------|
| `G` | 6.67428e-11 | Gravitational constant |
| `miu` | 0.5 | Friction coefficient |
| `fnsStifness` | 300 | Net cable stiffness |
| `fnsDamping` | 0.5 | Net cable damping |
| `SHlmax` | 25 | SH max storage degree |
| `SHdegree` | 21 | SH fitting degree |
| `Omega0` | 7.72e-4 rad/s | Bennu rotation rate |
| `OmegaTheta` | -78.6° | Bennu rotation axis θ |
| `OmegaPhi` | -39.4° | Bennu rotation axis φ |
| `start_of_control` | 0 | Control start time |

### Asteroid Rotation (Bennu)

```python
omega = 7.72e-4  # rad/s
T = 2 * np.pi / omega  # ~8138 s ≈ 2.26 hours
```

---

## 6. How These Feed Into the MDU Path Planning Project

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW PIPELINE                           │
└─────────────────────────────────────────────────────────────────────┘

Step 1: Load Net Topology
  └── FNS/*.txt ──→ Graph of N nodes + E edges
  └── This defines WHERE MDUs can move

Step 2: Load Final Net State
  └── Solution/last.txt ──→ Fixed 3D positions for all N nodes
  └── The net is now a static mesh wrapped around the asteroid

Step 3: Construct Traversal Graph
  └── Nodes = FNS.Pt (final positions from solution)
  └── Edges = FNS.Ed (connectivity unchanged)
  └── Edge weights = ||Pt[i] - Pt[j]|| (Euclidean distance on mesh)

Step 4: Load Asteroid Model
  └── PolyModel/*.txt ──→ Mesh for visual reference & coverage checking
  └── SHPara/*.txt ──→ Smooth surface for FOV intersection tests

Step 5: Define Cone FOV for Each MDU
  └── Each MDU at node position p with cone angle α, range r
  └── Coverage = set of asteroid surface points within cone
  └── Reward = uncovered area newly covered by this step

Step 6: DRL Environment
  └── State:  MDU positions on graph + covered mask + time
  └── Action: which neighbor node each MDU moves to
  └── Reward: +new_coverage, -time_penalty, -overlap_penalty
  └── Done:   coverage_threshold OR max_steps reached
```

---

## 7. MATLAB Codebase

**Location:** `C:\Users\Lenovo\Desktop\Study\AsteroidResearch\FNS_capture-IAA\matlab6_12\matlab6_12\`

The MATLAB code was the original simulation environment (later ported to C++).
Key files relevant to our project:

### Main Files

| File | Purpose |
|------|---------|
| `ShowAnimition.m` | **Core animation** — loads solution file + asteroid + net mesh, renders 3D animation frame by frame. Shows how solution data is parsed and visualized. |
| `GenerateMeshNodes.m` | **FNS mesh generator** — reads a simplified FNS file (just Np, Ne, Pt, Ed), subdivides long edges with intermediate nodes at `RopeLength` intervals, and builds full mesh with Ed connectivity. |
| `SurfacePoints.m` | **SH surface evaluator** — evaluates asteroid surface `R(θ, φ)`, returns Cartesian coords, tangents, and outward normal. |
| `SurfaceParameter.m` | **SH coefficient fitter** — fits spherical harmonics to polyhedron vertices. Solves least-squares to get Clm/Slm coefficients up to lmax=25. |
| `preload.m` | **Asteroid mesh loader** — loads `Asteroid_*.txt` format, computes edges, face normals, edge normals. Handles scaling. |
| `odeEuler.m` | **Euler integration** — time-stepping for net dynamics (same as C++ OdeEuler). |
| `SysDynEqn.m` | **System dynamics equations** — computes accelerations, gravity from polyhedron. |
| `calAcceltion.m` | Acceleration calculations. |
| `calcuGF.m` | Gravity field calculations. |
| `calExForce.m` / `calExConstraint.m` | External force / constraint calculations. |
| `calSurfaceForce.m` | Surface contact forces (net-asteroid interaction). |
| `SphericalCoordinates.m` | Coordinate conversion utilities. |

### MATLAB-Specific File Formats

**Asteroid file** (`Asteroid_bennu.txt`) — different from the C++ polyhedron format:
```
Line 1:    NumVerts NumFaces
Line 2..:  idx  x  y  z           (vertex index + coordinates, scientific notation)
Line ..:   idx  v1  v2  v3        (face index + 3 vertex indices)
```

Units: kilometers (divided by 1000 from meters). The `preload.m` function scales them
back to meters using xlong/ylong/zlong parameters.

Example:
```
        1348        2692
  1	0.00E+00	0.00E+00	2.53E-01    ← 0.253 km = 253 m
  2	2.80E-02	0.00E+00	2.63E-01    ← 0.028 km = 28 m
```

**FNS.matlab.txt** — simplified FNS format for MATLAB mesh generation:
```
Line 1:    Np Ne
Line 2..:  x  y  z               (node coordinates, one per line)
Line ..:  node_i node_j           (edge connectivity, 1-indexed!)
```
Note: node indices are **1-indexed** in MATLAB (unlike C++ which is 0-indexed).

### MATLAB Visualization Pipeline

From `ShowAnimition.m`:
1. Load `preload('Asteroid_bennu.txt')` → asteroid mesh + normals
2. `SurfaceParameter(p_data)` → fit SH coefficients from mesh vertices
3. `GenerateMeshNodes('FNS_square.matlab.txt')` → net mesh with subdivided edges
4. Load `NSGA2-demonstrate-*.txt` → trajectory solution
5. For each time step: reshape solution row into `Points(Np, 3)`, plot net edges + nodes + asteroid surface
6. Render: `surf()` for SH surface, `plot3()` lines for net edges

### Mesh Generation Detail

`GenerateMeshNodes.m` takes a coarse FNS file and subdivides edges:
- Each edge of length `L` gets `ceil(L/RopeLength) - 1` intermediate nodes placed linearly
- RopeLength = 5000 (default, in meters for C++; adjusted in MATLAB)
- Original mass nodes kept at end, new intermediate nodes inserted
- This is how the 369-node net is built from a much coarser definition

### Python Script

`cstyle2matlab.py` — Converts C-style FNS files (0-indexed, with topology columns) to
MATLAB format (1-indexed, simplified). Important: adds +1 to all node indices
because MATLAB is 1-indexed.

---

## 8. How These Feed Into the MDU Path Planning Project

For quick access, the key directories are:

```
# Net topology definitions
FNS_capture_control/FNS_capture/FNS_capture/FNS/
  └── FNS_square_fold-80m.txt          ← primary net topology

# Final net states (your starting point)
FNS_capture_control/FNS_capture/FNS_capture/Solution/
  └── last.txt                          ← final converged state

# Asteroid polyhedron meshes
FNS_capture_control/FNS_capture/FNS_capture/PolyModel/
  └── polyhedron_bennu.txt              ← Bennu mesh

# Spherical harmonics surface parameters
FNS_capture_control/FNS_capture/FNS_capture/SHPara/
  └── SHPara_bennu.txt                  ← Bennu SH coefficients

# Source code reference
FNS_capture_control/FNS_capture/FNS_capture/
  ├── LoadFNS.cpp/h                     ← FNS file parser
  ├── LoadPolyhedron.cpp/h              ← Polyhedron file parser
  ├── SurfacePoints.cpp/h               ← SH surface evaluation
  ├── OdeEuler.cpp                      ← Trajectory output format
  ├── trajectory.cpp/h                  ← Control logic reference
  └── constant.h                        ← Physical constants
```
