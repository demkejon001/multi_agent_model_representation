
RESULTS_DIRPATH="data/results/iterative_action/"
MODELS_DIRPATH="data/models/"

# Multi Strategy (4 agents) Char Embed: 64, Char Layer: 1, Mental Embed: 128, Mental Layer: 1
wandbLinks128=(jpd0057/tommas/model-uet8ojju:v0 jpd0057/tommas/model-2fq97yoj:v0 jpd0057/tommas/model-2elrmmcq:v0 jpd0057/tommas/model-11cy3ah8:v0 jpd0057/tommas/model-2zt8sz03:v0)
counter=1
for f in ${wandbLinks128[@]}; do
  python load.py "$f"
  python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
  mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[64,1]_lstm[128,1]_seed${counter}.ckpt
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[64,1]_lstm[128,1]_TrajectoryData[JAGrim4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[64,1]_lstm[128,1]_TrajectoryData[JAMirror4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[64,1]_lstm[128,1]_TrajectoryData[JAMixTgrPtrn4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[64,1]_lstm[128,1]_TrajectoryData[JAWSLS4]_seed${counter}.pickle
  ((counter++))
done

# Multi Strategy (4 agents) Char Embed: 64, Char Layer: 1, Mental Embed: 64, Mental Layer: 2
wandbLinks64=(jpd0057/tommas/model-2jp8aqg4:v0 jpd0057/tommas/model-fclhu1tz:v0 jpd0057/tommas/model-6kwri3sw:v0 jpd0057/tommas/model-1hyxij5g:v0 jpd0057/tommas/model-1m7lc8t3:v0)
counter=1
for f in ${wandbLinks64[@]}; do
  python load.py "$f"
  python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
  mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[64,1]_lstm[64,2]_seed${counter}.ckpt
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[64,1]_lstm[64,2]_TrajectoryData[JAGrim4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[64,1]_lstm[64,2]_TrajectoryData[JAMirror4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[64,1]_lstm[64,2]_TrajectoryData[JAMixTgrPtrn4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[64,1]_lstm[64,2]_TrajectoryData[JAWSLS4]_seed${counter}.pickle
  ((counter++))
done

# Multi Strategy (4 agents) Char Embed: 80, Char Layer: 1, Mental Embed: 64, Mental Layer: 1
wandbLinks64=(jpd0057/tommas/model-248s8q89:v0 jpd0057/tommas/model-8aa2cszl:v0 jpd0057/tommas/model-1l821hkp:v0 jpd0057/tommas/model-2019yrr2:v0 jpd0057/tommas/model-25fbmtlj:v0)
counter=1
for f in ${wandbLinks64[@]}; do
  python load.py "$f"
  python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
  mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[80,1]_lstm[64,1]_seed${counter}.ckpt
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[80,1]_lstm[64,1]_TrajectoryData[JAGrim4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[80,1]_lstm[64,1]_TrajectoryData[JAMirror4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[80,1]_lstm[64,1]_TrajectoryData[JAMixTgrPtrn4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[80,1]_lstm[64,1]_TrajectoryData[JAWSLS4]_seed${counter}.pickle
  ((counter++))
done

# Multi Strategy (4 agents) Char Embed: 80, Char Layer: 1, Mental Embed: 48, Mental Layer: 2
wandbLinks12848=(jpd0057/tommas/model-37rpr2kt:v0 jpd0057/tommas/model-3e7ru1we:v0 jpd0057/tommas/model-17yvptkq:v0 jpd0057/tommas/model-vveerzzu:v0 jpd0057/tommas/model-l8s07tlj:v0)
counter=1
for f in ${wandbLinks12848[@]}; do
  python load.py "$f"
  python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
  mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[80,1]_lstm[48,2]_seed${counter}.ckpt
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[80,1]_lstm[48,2]_TrajectoryData[JAGrim4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[80,1]_lstm[48,2]_TrajectoryData[JAMirror4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[80,1]_lstm[48,2]_TrajectoryData[JAMixTgrPtrn4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[80,1]_lstm[48,2]_TrajectoryData[JAWSLS4]_seed${counter}.pickle
  ((counter++))
done

# Multi Strategy (4 agents) Char Embed: 128, Char Layer: 1, Mental Embed: 48, Mental Layer: 2
wandbLinks12848=(jpd0057/tommas/model-385vgbmz:v0 jpd0057/tommas/model-1553o7j0:v0 jpd0057/tommas/model-19ab1xzq:v0 jpd0057/tommas/model-ri21thpc:v0 jpd0057/tommas/model-1c7602rm:v0)
counter=1
for f in ${wandbLinks12848[@]}; do
  python load.py "$f"
  python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
  mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[128,1]_lstm[48,2]_seed${counter}.ckpt
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[128,1]_lstm[48,2]_TrajectoryData[JAGrim4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[128,1]_lstm[48,2]_TrajectoryData[JAMirror4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[128,1]_lstm[48,2]_TrajectoryData[JAMixTgrPtrn4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[128,1]_lstm[48,2]_TrajectoryData[JAWSLS4]_seed${counter}.pickle
  ((counter++))
done

# Multi Strategy (4 agents) Char Embed: 256, Char Layer: 1, Mental Embed: 48, Mental Layer: 2
wandbLinks12848=(jpd0057/tommas/model-hj6gp4cc:v0 jpd0057/tommas/model-9lpphrhn:v0 jpd0057/tommas/model-1afqqk18:v0 jpd0057/tommas/model-172l9cy4:v0 jpd0057/tommas/model-3hllgeht:v0)
counter=1
for f in ${wandbLinks12848[@]}; do
  python load.py "$f"
  python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
  mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[256,1]_lstm[48,2]_seed${counter}.ckpt
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[256,1]_lstm[48,2]_TrajectoryData[JAGrim4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[256,1]_lstm[48,2]_TrajectoryData[JAMirror4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[256,1]_lstm[48,2]_TrajectoryData[JAMixTgrPtrn4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[256,1]_lstm[48,2]_TrajectoryData[JAWSLS4]_seed${counter}.pickle
  ((counter++))
done

#Multi Strategy (4 agents) Char Embed: 64, Char Layer: 1, Mental Embed: 48, Mental Layer: 2
python load.py jpd0057/tommas/model-xb693mer:v0
python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[64,1]_lstm[48,2].ckpt
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[64,1]_lstm[48,2]_TrajectoryData[JAGrim4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[64,1]_lstm[48,2]_TrajectoryData[JAMirror4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[64,1]_lstm[48,2]_TrajectoryData[JAMixTgrPtrn4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[64,1]_lstm[48,2]_TrajectoryData[JAWSLS4].pickle

#Multi Strategy (4 agents) Char Embed: 80, Char Layer: 2, Mental Embed: 64, Mental Layer: 1
python load.py jpd0057/tommas/model-1q9spnix:v0
python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[80,2]_lstm[64,1].ckpt
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[80,2]_lstm[64,1]_TrajectoryData[JAGrim4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[80,2]_lstm[64,1]_TrajectoryData[JAMirror4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[80,2]_lstm[64,1]_TrajectoryData[JAMixTgrPtrn4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[80,2]_lstm[64,1]_TrajectoryData[JAWSLS4].pickle

#Multi Strategy (4 agents) Char Embed: 128, Char Layer: 1, Mental Embed: 48, Mental Layer: 1
python load.py jpd0057/tommas/model-1zlaj4fg:v0
python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[128,1]_lstm[48,1].ckpt
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[128,1]_lstm[48,1]_TrajectoryData[JAGrim4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[128,1]_lstm[48,1]_TrajectoryData[JAMirror4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[128,1]_lstm[48,1]_TrajectoryData[JAMixTgrPtrn4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[128,1]_lstm[48,1]_TrajectoryData[JAWSLS4].pickle

#Multi Strategy (4 agents) Char Embed: 256, Char Layer: 1, Mental Embed: 48, Mental Layer: 2
python load.py jpd0057/tommas/model-2txezupw:v0
python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[256,1]_lstm[48,2].ckpt
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[256,1]_lstm[48,2]_TrajectoryData[JAGrim4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[256,1]_lstm[48,2]_TrajectoryData[JAMirror4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[256,1]_lstm[48,2]_TrajectoryData[JAMixTgrPtrn4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[256,1]_lstm[48,2]_TrajectoryData[JAWSLS4].pickle

# Multi Strategy (4 agents) Char Embed: 128, Char Layer: 2, Mental Embed: 48, Mental Layer: 2
python load.py jpd0057/tommas/model-xf5du4ai:v0
python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[128,2]_lstm[48,2].ckpt
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[128,2]_lstm[48,2]_TrajectoryData[JAGrim4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[128,2]_lstm[48,2]_TrajectoryData[JAMirror4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[128,2]_lstm[48,2]_TrajectoryData[JAMixTgrPtrn4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[128,2]_lstm[48,2]_TrajectoryData[JAWSLS4].pickle

# Multi Strategy (4 agents) Char Embed: 512, Char Layer: 1, Mental Embed: 48, Mental Layer: 2
python load.py jpd0057/tommas/model-1dkxn11x:v0
python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[512,1]_lstm[48,2].ckpt
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[512,1]_lstm[48,2]_TrajectoryData[JAGrim4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[512,1]_lstm[48,2]_TrajectoryData[JAMirror4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[512,1]_lstm[48,2]_TrajectoryData[JAMixTgrPtrn4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_lstm[512,1]_lstm[48,2]_TrajectoryData[JAWSLS4].pickle