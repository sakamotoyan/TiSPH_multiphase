
# TiSPH_multiphase
TiSPH_multiphase is an open-source implementation of our implicit multiphase solver.

## Platform
This software is developed using [Taichi](https://docs.taichi.graphics/), making it compatible with both Linux and Windows operating systems, with or without a graphics card. It has been thoroughly tested on Ubuntu 20.04 and Windows 10/11.

For an optimal experience, especially when running demos, we recommend using an operating system with a graphical user interface (GUI).

## Requirements
- **Taichi**: The core of most algorithms in this program. Install it via pip with the command:
  ```bash
  pip install taichi
  ```
- **Numpy**: Essential for data generation.
- **Matplotlib**: Used for visualizing and plotting results.
- **OpenGL or Vulkan**: Necessary for the GUI-based 3D demos.

## Demos
### Running the demos

1. Create a dedicated folder named `output` within the program's directory. This folder will be used to store the results produced by the program..

2. Execute the following in your terminal to run a demo:
```bash
python scene_xxx.py
```
**2D Demos**: The output consists of a series of images that are automatically saved to the `output` directory.

**3D Demos**: A window will open displaying the simulation. Use your mouse to adjust the camera angle. Press `r` to start the simulation.

### GPU Simulation
By default, all 2D demos are set to run on the CPU. To run simulations on the GPU:
1. Comment out `ti.init(arch=ti.cpu)` 
2. Uncomment `ti.init(arch=ti.cuda)` in the `scene_xxx.py` file.

> **Note**: Ensure you have a CUDA-compatible graphics card for GPU simulation. We've verified performance on the NVIDIA RTX 3090. The demo `scene_3D_multiphase_separate.py` is a time-intensive 3D simulation and is set to run on the GPU by default.

### Solver Configuration
All demos utilize our implicit multiphase solver by default. To switch to another multiphase solver:
1. Change `solver = SOLVER_ISM` 
2. To `solver = SOLVER_JL21` in the `scene_xxx.py` file.

### Experimenting with Demos
For those keen to tweak and experiment, we've included ample comments within the `scene_xxx.py` file. Adjust parameters as you like and observe varying results!
