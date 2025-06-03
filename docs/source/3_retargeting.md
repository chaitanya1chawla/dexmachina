# Kinematic Retargeting

We require kinematic retargeting for every combination of dexterous hand and human hand-object demonstration clip. The same procedure is applied to all existing tasks and hands, and documented below for adding new assets. 

```{note}
Additionally need to install [dex-retargeting](https://github.com/dexsuite/dex-retargeting):`pip install dex_retargeting`. 
```  




## Hand Config Preparation 
For each new pair of dexterous hands, you would need to add its URDF and mesh assets into `assets` folder, and manually add a new `assets/hand_folder/retarget_config.yaml` file, which specifies the desired fingertip mapping from dexterous hand keypoints to human hands (MANO hands). See more information in {doc}`1_process_hands`. 

## Collision-aware Physics-enabled Retargeting

