
RESULTS_DIRPATH="data/results/iterative_action/"
MODELS_DIRPATH="data/models/"
# Multi Strategy (4 agents) Char Embed: 48, Char Layer: 6, Char Head: 2, Mental Embed: 48, Mental Layer: 8, Mental Head: 4
wandbLinks48=(jpd0057/tommas/model-2liyjuqf:v0 jpd0057/tommas/model-mh3e8byq:v0 jpd0057/tommas/model-3qzqkp42:v0 jpd0057/tommas/model-1iw6b4i2:v0 jpd0057/tommas/model-kqbi5gc0:v0)
seeds48=(1 2 3 4 5)
for i in "${!wandbLinks48[@]}"; do
  python load.py "${wandbLinks48[i]}"
  counter="${seeds48[i]}"
  python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
  mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[48,6,2]_ttx[48,8,4]_seed${counter}.ckpt
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[48,6,2]_ttx[48,8,4]_TrajectoryData[JAGrim4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[48,6,2]_ttx[48,8,4]_TrajectoryData[JAMirror4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[48,6,2]_ttx[48,8,4]_TrajectoryData[JAMixTgrPtrn4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[48,6,2]_ttx[48,8,4]_TrajectoryData[JAWSLS4]_seed${counter}.pickle
done

# Multi Strategy (4 agents) Char Embed: 64, Char Layer: 8, Char Head: 4, Mental Embed: 48, Mental Layer: 8, Mental Head: 4
wandbLinks64=(jpd0057/tommas/model-rv8f3jsf:v0 jpd0057/tommas/model-1hu95hdd:v0 jpd0057/tommas/model-gjuivdh7:v0 jpd0057/tommas/model-g5ak567t:v0 jpd0057/tommas/model-1bo6d6xq:v0)
seeds64=(1 2 3 4 5)
for i in ${!wandbLinks64[@]}; do
  python load.py "${wandbLinks64[i]}"
  counter="${seeds64[i]}"
  python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
  mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[64,8,4]_ttx[48,8,4]_seed${counter}.ckpt
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[64,8,4]_ttx[48,8,4]_TrajectoryData[JAGrim4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[64,8,4]_ttx[48,8,4]_TrajectoryData[JAMirror4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[64,8,4]_ttx[48,8,4]_TrajectoryData[JAMixTgrPtrn4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[64,8,4]_ttx[48,8,4]_TrajectoryData[JAWSLS4]_seed${counter}.pickle
done

# Multi Strategy (4 agents) Char Embed: 128, Char Layer: 2, Char Head: 4, Mental Embed: 48, Mental Layer: 8, Mental Head: 4
wandbLinks64=(jpd0057/tommas/model-rv8f3jsf:v0 jpd0057/tommas/model-1hu95hdd:v0 jpd0057/tommas/model-gjuivdh7:v0 jpd0057/tommas/model-g5ak567t:v0 jpd0057/tommas/model-1bo6d6xq:v0)
seeds64=(1 2 3 4 5)
for i in ${!wandbLinks64[@]}; do
  python load.py "${wandbLinks64[i]}"
  counter="${seeds64[i]}"
  python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
  mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[128,2,4]_ttx[48,8,4]_seed${counter}.ckpt
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[128,2,4]_ttx[48,8,4]_TrajectoryData[JAGrim4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[128,2,4]_ttx[48,8,4]_TrajectoryData[JAMirror4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[128,2,4]_ttx[48,8,4]_TrajectoryData[JAMixTgrPtrn4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[128,2,4]_ttx[48,8,4]_TrajectoryData[JAWSLS4]_seed${counter}.pickle
done

# Multi Strategy (4 agents) Char Embed: 80, Char Layer: 4, Char Head: 4, Mental Embed: 64, Mental Layer: 4, Mental Head: 4
wandbLinks64=(jpd0057/tommas/model-2g6suign:v0 jpd0057/tommas/model-2p0fa6cx:v0 jpd0057/tommas/model-1mgz91nv:v0 jpd0057/tommas/model-dqw7x36h:v0 jpd0057/tommas/model-3rjvzfpu:v0)
seeds64=(1 2 3 4 5)
for i in ${!wandbLinks64[@]}; do
  python load.py "${wandbLinks64[i]}"
  counter="${seeds64[i]}"
  python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
  mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[80,4,4]_ttx[64,4,4]_seed${counter}.ckpt
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[80,4,4]_ttx[64,4,4]_TrajectoryData[JAGrim4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[80,4,4]_ttx[64,4,4]_TrajectoryData[JAMirror4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[80,4,4]_ttx[64,4,4]_TrajectoryData[JAMixTgrPtrn4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[80,4,4]_ttx[64,4,4]_TrajectoryData[JAWSLS4]_seed${counter}.pickle
done

# Multi Strategy (4 agents) Char Embed: 48, Char Layer: 8, Char Head: 4, Mental Embed: 64, Mental Layer: 4, Mental Head: 4
wandbLinks64=(jpd0057/tommas/model-26r6volf:v0 jpd0057/tommas/model-3jpwusnm:v0 jpd0057/tommas/model-2any7s78:v0 jpd0057/tommas/model-nn3ilh7z:v0 jpd0057/tommas/model-2ra3akae:v0)
seeds64=(1 2 3 4 5)
for i in ${!wandbLinks64[@]}; do
  python load.py "${wandbLinks64[i]}"
  counter="${seeds64[i]}"
  python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
  mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[48,8,4]_ttx[64,4,4]_seed${counter}.ckpt
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[48,8,4]_ttx[64,4,4]_TrajectoryData[JAGrim4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[48,8,4]_ttx[64,4,4]_TrajectoryData[JAMirror4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[48,8,4]_ttx[64,4,4]_TrajectoryData[JAMixTgrPtrn4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[48,8,4]_ttx[64,4,4]_TrajectoryData[JAWSLS4]_seed${counter}.pickle
done

# Multi Strategy (4 agents) Char Embed: 128, Char Layer: 4, Char Head: 8, Mental Embed: 64, Mental Layer: 4, Mental Head: 4
wandbLinks64=(jpd0057/tommas/model-1rocnyqb:v0 jpd0057/tommas/model-2dumm0e9:v0 jpd0057/tommas/model-3ad9g0b5:v0 jpd0057/tommas/model-2osn94mb:v0 jpd0057/tommas/model-3qyv2ovr:v0)
seeds64=(1 2 3 4 5)
for i in ${!wandbLinks64[@]}; do
  python load.py "${wandbLinks64[i]}"
  counter="${seeds64[i]}"
  python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
  mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[128,4,8]_ttx[64,4,4]_seed${counter}.ckpt
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[128,4,8]_ttx[64,4,4]_TrajectoryData[JAGrim4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[128,4,8]_ttx[64,4,4]_TrajectoryData[JAMirror4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[128,4,8]_ttx[64,4,4]_TrajectoryData[JAMixTgrPtrn4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[128,4,8]_ttx[64,4,4]_TrajectoryData[JAWSLS4]_seed${counter}.pickle
done

# Multi Strategy (4 agents) Char Embed: 64, Char Layer: 4, Char Head: 4, Mental Embed: 64, Mental Layer: 4, Mental Head: 4
wandbLinks64=(jpd0057/tommas/model-dl3ds106:v0 jpd0057/tommas/model-zvknwgzj:v0 jpd0057/tommas/model-34j5xxcl:v0 jpd0057/tommas/model-2cyzl2dr:v0 jpd0057/tommas/model-1m76wjwd:v0)
seeds64=(1 2 3 4 5)
for i in ${!wandbLinks64[@]}; do
  python load.py "${wandbLinks64[i]}"
  counter="${seeds64[i]}"
  python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
  mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[64,4,4]_ttx[64,4,4]_seed${counter}.ckpt
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[64,4,4]_ttx[64,4,4]_TrajectoryData[JAGrim4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[64,4,4]_ttx[64,4,4]_TrajectoryData[JAMirror4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[64,4,4]_ttx[64,4,4]_TrajectoryData[JAMixTgrPtrn4]_seed${counter}.pickle
  mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[64,4,4]_ttx[64,4,4]_TrajectoryData[JAWSLS4]_seed${counter}.pickle
done

#Multi Strategy (4 agents) Ec: 32, Lc: 4, Hc: 2, Em: 48, Lm: 8, Hm: 4
python load.py jpd0057/tommas/model-hxpek9nf:v0
python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[32,4,2]_ttx[48,8,4].ckpt
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[32,4,2]_ttx[48,8,4]_TrajectoryData[JAGrim4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[32,4,2]_ttx[48,8,4]_TrajectoryData[JAMirror4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[32,4,2]_ttx[48,8,4]_TrajectoryData[JAMixTgrPtrn4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[32,4,2]_ttx[48,8,4]_TrajectoryData[JAWSLS4].pickle

#Multi Strategy (4 agents) Ec: 128, Lc: 8, Hc: 2, Em: 48, Lm: 8, Hm: 4
python load.py jpd0057/tommas/model-1chb1gcm:v0
python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[128,8,2]_ttx[48,8,4].ckpt
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[128,8,2]_ttx[48,8,4]_TrajectoryData[JAGrim4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[128,8,2]_ttx[48,8,4]_TrajectoryData[JAMirror4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[128,8,2]_ttx[48,8,4]_TrajectoryData[JAMixTgrPtrn4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[128,8,2]_ttx[48,8,4]_TrajectoryData[JAWSLS4].pickle

#Multi Strategy (4 agents) Ec: 64, Lc: 8, Hc: 4, Em: 128, Lm: 2, Hm: 8
python load.py jpd0057/tommas/model-1dd9qhxq:v0
python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 4
mv ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4].ckpt ${MODELS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[64,8,4]_ttx[128,2,8].ckpt
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAGrim4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[64,8,4]_ttx[128,2,8]_TrajectoryData[JAGrim4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMirror4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[64,8,4]_ttx[128,2,8]_TrajectoryData[JAMirror4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAMixTgrPtrn4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[64,8,4]_ttx[128,2,8]_TrajectoryData[JAMixTgrPtrn4].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_TrajectoryData[JAWSLS4].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4]_ttx[64,8,4]_ttx[128,2,8]_TrajectoryData[JAWSLS4].pickle
