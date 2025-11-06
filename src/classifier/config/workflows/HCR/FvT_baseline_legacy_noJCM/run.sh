# TODO: this should be removed after new skim 
export LPCUSER="chuyuanl"
export CERNUSER="c/chuyuan"
export WFS="classifier/config/workflows/HCR/FvT_baseline_legacy"
export BASE="root://cmseos.fnal.gov//store/user/${LPCUSER}/HH4b"
export MODEL="${BASE}/classifier/FvT/baseline_legacy/noJCM/"
export WEB="root://eosuser.cern.ch//eos/user/${CERNUSER}/www/HH4b/classifier/FvT_baseline_legacy/noJCM/"
export GMAIL=~/gmail.yml

# check port
if [ -z "$1" ]; then
    port=10200
else
    port=$1
fi

# train mixed and make plots
./pyml.py template "{offset: 0, user: ${LPCUSER}}" $WFS/train.yml -setting Monitor "address: :${port}"
./pyml.py analyze --results ${MODEL}/data/result.json -analysis HCR.LossROC -setting IO "output: ${WEB}" -setting IO "report: data" -setting Monitor "address: :${port}"
# evaluate
./pyml.py template "{user: ${LPCUSER}}" $WFS/evaluate.yml -setting Monitor "address: :${port}"

if [ -e "$GMAIL" ]; then
    ./pyml.py analyze \
        -analysis notify.Gmail \
        --title "FvT data baseline done" \
        --body "finished at $(date)" \
        --labels Classifier HH4b \
        -from $GMAIL \
        -setting Monitor "address: :${port}"
fi