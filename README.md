### Training Cap
- sh run_caption.sh

### Training Proposal,Separately
- sh run_proposal.sh

### Training Proposal,Joint
- pretrained_cap_model_path=./checkpoint/train_cap/对应的权重文件名/best_cap_model.pt
- 并在run_proposal.sh命令行中添加对应的参数
- sh run_proposal.sh

### Learned Proposal
- pretrained_cap_path=./checkpoint/train_cap/对应的caption文件名/best_cap_model.pt
- prop_result_path=./log/train_prop/对应的proposal文件名/submissions/prop_results_val_1_e最优周期_maxprop100.json
- sh run_eval_props.sh

### 注:以下文件可以通过软链接实现
- data             链接CAPTION_DATA
- submodules       链接submodules
- .vector_cache    链接.vector_cache   
 
