# TODO: this should be removed after new skim 
export LPCUSER="chuyuanl"
export CERNUSER="c/chuyuan"
export WFS="classifier/config/workflows/HCR/FvT_baseline_legacy"
export BASE="root://cmseos.fnal.gov//store/user/${LPCUSER}/HH4b"
export MODEL="${BASE}/classifier/FvT/baseline_legacy/JCM/"
export WEB="root://eosuser.cern.ch//eos/user/${CERNUSER}/www/HH4b/classifier/FvT_baseline_legacy/JCM/"
export GMAIL=~/gmail.yml

# check port
if [ -z "$1" ]; then
    port=10200
else
    port=$1
fi

# train mixed and make plots
for i in {0..14}
do
    ./pyml.py template "{mixed: ${i}, offset: 0, user: ${LPCUSER}}" $WFS/train_mixed.yml -setting Monitor "address: :${port}"
    ./pyml.py analyze --results ${MODEL}/mixed-${i}/result.json -analysis HCR.LossROC -setting IO "output: ${WEB}" -setting IO "report: mixed-${i}" -setting Monitor "address: :${port}"
done
# evaluate
for i in {0..14}
do
    ./pyml.py template "{mixed: ${i}, user: ${LPCUSER}}" $WFS/evaluate_mixed.yml -setting Monitor "address: :${port}"
done

if [ -e "$GMAIL" ]; then
    ./pyml.py analyze \
        -analysis notify.Gmail \
        --title "FvT mixed baseline done" \
        --body "finished at $(date)" \
        --labels Classifier HH4b \
        -from $GMAIL \
        -setting Monitor "address: :${port}"
fi