# multi_agent_model_representation
Evaluating Multi-Agent Modellers' Representations

# Install
Use python=3.6

`pip install -r requirements.txt`

# Create Datasets
Iterative Action Dataset:  
```console
python create_datasets.py --action_mirror_ja4 --action_wsls_ja4 --action_grim_trigger_ja4 --action_mixed_trigger_pattern_ja
```

Gridworld Dataset:  
```console
python create_datasets.py --gridworld
```

# Training
Some example commands used to train the modellers.  
Iterative Action:
```console
python train.py -d "training for iterative action analysis" --model iterative_action_tomnet --dataset action_mirror_ja4 action_wsls_ja4 action_grim_trigger_ja4 action_mixed_trigger_pattern_ja4 --learning_rate .001 --num_minibatches 80000 --n_past 5 --batch_size 64 --log_eval_interval 250 --log_train_interval 50 --seed 1 --lstm_char --lstm_mental --char_embedding_size 512 --char_n_layer 2 --mental_embedding_size 64 --mental_n_layer 2
```

Gridworld:
```console
python train.py -d "training for gridworld analysis" --model gridworld_tomnet --dataset gridworld --learning_rate .0001 --num_minibatches 40000 --n_past 5 --batch_size 32 --accumulate_grad_batches 1 --log_eval_interval 250 --log_train_interval 50 --seed 1 --char_embedding_size 64 --char_n_layer 8 --char_n_head 8 --mental_embedding_size 64 --mental_n_layer 4 --mental_n_head 8 --pred_net_features 64 --gridworld_embedding_size 64 --action_embedding_size 8 --no_sr
```

The `--lstm_char --lstm_mental` arguments will make the ToMnet modellers use LSTMs

Additionally you can add:  
`--model_checkpointing`: to save your models  
`--logging`: To log your results to [WandB](wandb.ai)  
`--log_model`: To log your model to [WandB](wandb.ai)

# Analysis
If you use WandB model logging then you can add your artifact links to `load_iterative_action_artifacts.py` or `load_gridworld_artifacts.py` and download your trained modellers.
`src.analysis.iterative_action_representation_metrics` and `src.analysis.gridworld_representation_metrics` contain functions for analyzing the modeller's representations and each module has a `collate_and_save_all_representation_data()` which will run the representation analysis and save the data to pandas dataframes.
