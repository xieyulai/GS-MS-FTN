# GS-MS-FTN for MM-DVC
Code and data for the paper [Global-shared Text Representation based Multi-Stage Fusion Transformer Network for Multi-modal Dense Video Captioning](https://ieeexplore.ieee.org/document/10227555/)

NOTE: This repo is still under construction.


Please cite the following paper if you use this repository in your research.
```
@ARTICLE{10227555,
  author={Xie, Yulai and Niu, Jingjing and Zhang, Yang and Ren, Fang},
  journal={IEEE Transactions on Multimedia}, 
  title={Global-Shared Text Representation Based Multi-Stage Fusion Transformer Network for Multi-Modal Dense Video Captioning}, 
  year={2024},
  volume={26},
  number={},
  pages={3164-3179},
  keywords={Proposals;Visualization;Task analysis;Semantics;Transformers;Correlation;Fuses;Anchor-free target detection;dense video captioning;global-shared text;multi-modal analysis;multi-stage fusion},
  doi={10.1109/TMM.2023.3307972}}

```

### Training Cap
- sh run_caption.sh

```
#configure the training
DATA_SELECT=2000 # SELECT 2000 or 9000
```

### Training Proposal
- sh run_proposal.sh


### Learned Proposal
- pretrained_cap_path=./checkpoint/train_cap/{caption_file_path}/best_cap_model.pt
- prop_result_path=./log/train_prop/{propsal_file_path}/submissions/prop_results_val_1_e{best_epoch}_maxprop100.json
- sh run_eval_props.sh

 
