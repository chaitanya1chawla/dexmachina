# Installation

## Minimal Requirements
1. We recommend using conda environment with Python=3.10
```
conda create -n dexmachina python=3.10
conda activate dexmachina
```
2. Clone and install this custom fork version of Genesis (a modified version that supports entity-to-entity contact position reading, disable default visualizer, group-based collision filtering, etc) 
```
pip install torch==2.2.2 
git clone https://github.com/MandiZhao/Genesis.git
cd Genesis
pip install -e .
```

3. Install this custom version of rl-games, which supports wandb logging and curriculum setting.
```
git clone https://github.com/MandiZhao/rl_games.git
cd rl_games
pip install -e .
```

## Additional Package Dependencies 

1. Kinematic retargeting
Install the [dex-retargeting](https://github.com/dexsuite/dex-retargeting) package:`pip install dex_retargeting`. 
2. Process Additional ARCTIC data 
3. Raytracing rendering 
- Follow the official instruction to build this separate raytracer package: [here](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/visualization.html#photo-realistic-ray-tracing-rendering)
- You might need sudo install a new cuda driver globally: Try `wget` to install [this link](https://developer.nvidia.com/cuda-12-2-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) -- it installs the latest driver 570 and cuda12.8
