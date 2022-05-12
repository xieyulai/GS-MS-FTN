#!/bin/bash

PROCEDURE=evaluate

MODALITY=audio_video_text

pretrained_cap_path=./checkpoint/train_cap/0325110315_audio_video_text_2000/best_cap_model.pt
prop_result_path=./log/train_prop/C1.0_0402105342_audio_video_text/submissions/prop_results_val_1_e17_maxprop100.json

BATCH=16

DEVICE_IDS=0

python main_fcos.py --procedure $PROCEDURE  --modality $MODALITY\
            --pretrained_cap_model_path $pretrained_cap_path\
            --prop_pred_path $prop_result_path\
            --device_ids $DEVICE_IDS --B $BATCH

