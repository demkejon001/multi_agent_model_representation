
RESULTS_DIRPATH="data/results/iterative_action/"

# Make sure WandB links are in order of TTX+TTX, LSTM+TTX, TTX+LSTM, LSTM+LSTM
wandbLinks=(jpd0057/tommas/model-16oky7k9:v0 jpd0057/tommas/model-130kd9uv:v0 jpd0057/tommas/model-mi85e016:v0 jpd0057/tommas/model-3voigddm:v0) # seed 314
#wandbLinks=(jpd0057/tommas/model-2kve5lbc:v0 jpd0057/tommas/model-21hh025i:v0 jpd0057/tommas/model-z2o21l5b:v0 jpd0057/tommas/model-3uwnxxmq:v0) # seed 42

python load.py ${wandbLinks[0]} # TTX+TTX
python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 2
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_TrajectoryData[JAGrim2].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_ttx_ttx_TrajectoryData[JAGrim2].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_TrajectoryData[JAMirror2].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_ttx_ttx_TrajectoryData[JAMirror2].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_TrajectoryData[JAMixTgrPtrn2].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_ttx_ttx_TrajectoryData[JAMixTgrPtrn2].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_TrajectoryData[JAWSLS2].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_ttx_ttx_TrajectoryData[JAWSLS2].pickle

python load.py ${wandbLinks[1]} # LSTM+TTX
python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 2
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_TrajectoryData[JAGrim2].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_lstm_ttx_TrajectoryData[JAGrim2].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_TrajectoryData[JAMirror2].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_lstm_ttx_TrajectoryData[JAMirror2].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_TrajectoryData[JAMixTgrPtrn2].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_lstm_ttx_TrajectoryData[JAMixTgrPtrn2].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_TrajectoryData[JAWSLS2].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_lstm_ttx_TrajectoryData[JAWSLS2].pickle

python load.py ${wandbLinks[2]} # TTX+LSTM
python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 2
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_TrajectoryData[JAGrim2].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_ttx_lstm_TrajectoryData[JAGrim2].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_TrajectoryData[JAMirror2].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_ttx_lstm_TrajectoryData[JAMirror2].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_TrajectoryData[JAMixTgrPtrn2].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_ttx_lstm_TrajectoryData[JAMixTgrPtrn2].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_TrajectoryData[JAWSLS2].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_ttx_lstm_TrajectoryData[JAWSLS2].pickle

python load.py ${wandbLinks[3]} # LSTM+LSTM
python iterative_action_results_compilation.py --trained_model iterative_action_past_current --trained_on multi_strat --results_on multi_strat --num_agents 2
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_TrajectoryData[JAGrim2].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_lstm_lstm_TrajectoryData[JAGrim2].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_TrajectoryData[JAMirror2].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_lstm_lstm_TrajectoryData[JAMirror2].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_TrajectoryData[JAMixTgrPtrn2].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_lstm_lstm_TrajectoryData[JAMixTgrPtrn2].pickle
mv ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_TrajectoryData[JAWSLS2].pickle ${RESULTS_DIRPATH}IterPastCur[JAGrim2,JAMirror2,JAMixTgrPtrn2,JAWSLS2]_lstm_lstm_TrajectoryData[JAWSLS2].pickle
