# ARCTIC Human Demonstration Data Processing 

## Process raw ARCTIC data
If you want to add a new object/demonstration clip in order to make a new task environment, you would need to process more data from ARCTIC. 

First, follow the installation instructions from the [ARCTIC](https://github.com/zc-alexfan/arctic) repo (preferablly in a separate conda environment) 

To download raw ARCTIC data, use their bash scripts: `./bash/download_misc.sh` and `./bash/download_body_models.sh` 
` 
The raw data you downloaded from ARCTIC will contain only the essential data, which needs to be further processed to generate per-step object and MANO hand mesh (we need this for contact estimatation). Inside the `arctic` repo, you would need to run something like:
```
python scripts_data/process_seqs.py --mano_p downloads/data/raw_seqs/s01/microwave_use_02.mano.npy --export_verts
```
which will generate a full sequence named `outputs/processed_verts/seqs/s01/microwave_use_02.npy`. ARCTIC has a nice built-in viewer that can load the demonstration and visualize them locally, the command looks something like:
```
python scripts_data/visualizer.py --no_image --seq_p arctic/outputs/processed_verts/seqs/s01/microwave_use_02.npy --mano --object
```

## Contact Approximation
Then, come back to this `dexmachina` repo and run `process_arctic.py` to further process the output demonstration sequences.

## Convex Decomposition on Collision Mesh
