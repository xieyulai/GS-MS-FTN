# GS-MS-FTN for MM-DVC
Code and data for the paper [Global-shared Text Representation based Multi-Stage Fusion Transformer Network for Multi-modal Dense Video Captioning]()

NOTE: This repo is still under construction.


Please cite the following paper if you use this repository in your research.
```
Under construction
```

### Training Cap
- sh run_caption.sh
'''
DATA_SELECT=2000 # SELECT 2000 or 9000
'''

### Training Proposal
- sh run_proposal.sh


### Learned Proposal
- pretrained_cap_path=./checkpoint/train_cap/{caption_file_path}/best_cap_model.pt
- prop_result_path=./log/train_prop/{propsal_file_path}/submissions/prop_results_val_1_e{best_file_with_epoch}_maxprop100.json
- sh run_eval_props.sh

 
