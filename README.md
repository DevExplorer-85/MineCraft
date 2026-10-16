
# VoxelWorld

A Python-based 3D voxel game featuring procedural terrain generation.

## Features

- Procedural terrain generation using OpenSimplex noise
- Chunk-based voxel world
- Pygame and ModernGL rendering
- Numba-accelerated terrain generation
- Grass, dirt, stone, sand, and snow blocks
- Experimental mountain ranges and desert valley generation

## Tech Stack

- Python
- Pygame
- ModernGL
- NumPy
- Numba
- OpenSimplex
- PyGLM

## Installation

1. Install Python.
2. Clone this repository.
3. Create a virtual environment.
4. Install the dependencies.
5. Run the game's entry-point script.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the game using its actual entry-point file, for example:

```powershell
python main.py
```

## Project Status

Work in progress. Terrain generation, materials, and world generation are being developed.

## License

No license has been selected yet.
