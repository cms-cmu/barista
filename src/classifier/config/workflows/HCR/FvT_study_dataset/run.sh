# TODO: this should be removed after new skim 
export LPCUSER="chuyuanl"
export CERNUSER="c/chuyuan"
export WFS="classifier/config/workflows/HCR/FvT_study_dataset"
export BASE="root://cmseos.fnal.gov//store/user/${LPCUSER}/HH4b"
export MODEL="${BASE}/classifier/FvT/$1"
export WEB="root://eosuser.cern.ch//eos/user/${CERNUSER}/www/HH4b/classifier/FvT_study/$1"
export GMAIL=~/gmail.yml

# check port
if [ -z "$2" ]; then
    port=10200
else
    port=$2
fi


# train mixed and make plots
for i in {0..14}
do
    ./pyml.py \
    template "{mixed: ${i}, user: ${LPCUSER}, name: $1, offset: 0}" $WFS/train.yml \
    -template "{mixed: ${i}, user: ${LPCUSER}}" $WFS/train_$1.yml \
    -setting Monitor "address: :${port}"

    ./pyml.py analyze --results ${MODEL}/mixed-${i}/result.json \
    -analysis HCR.LossROC \
    -setting IO "output: ${WEB}" \
    -setting IO "report: mixed-${i}" \
    -setting Monitor "address: :${port}"
done
# evaluate
for i in {0..14}
do
    ./pyml.py \
    template "{mixed: ${i}, user: ${LPCUSER}, name: $1}" $WFS/evaluate.yml \
    -setting Monitor "address: :${port}"
done

if [ -e "$GMAIL" ]; then
    ./pyml.py analyze \
        -analysis notify.Gmail \
        --title "FvT dataset study done" \
        --body "${1} finished at $(date)" \
        --labels Classifier HH4b \
        -from $GMAIL \
        -setting Monitor "address: :${port}"
fi