# change these vars #
export LPCUSER="chuyuanl"
export CERNUSER="c/chuyuan"
export BASE="root://cmseos.fnal.gov//store/user/${LPCUSER}/HH4b_2024_v2"
export MODEL="${BASE}/classifier/SvB/"
export SvB="${BASE}/friend/SvB/"
export FvT="${BASE}/friend/FvT/"
export PLOT="root://eosuser.cern.ch//eos/user/${CERNUSER}/www/HH4b/classifier/HH4b_2024_v2/"
#####################

export WFS="classifier/config/workflows/HH4b_2024_v2/SvB"

# check port
if [ -z "$1" ]; then
    port=10200
else
    port=$1
fi


# train
./pyml.py \
    template "{model: ${MODEL}, FvT: ${FvT}}" $WFS/train.yml \
    -from $WFS/../common.yml \
    -setting Monitor "address: :${port}" -flag debug

# plot the AUC and ROC
./pyml.py analyze \
    --results ${MODEL}/result.json \
    -analysis HCR.LossROC \
    -setting IO "output: ${PLOT}" \
    -setting IO "report: SvB" \
    -setting Monitor "address: :${port}"

# evaluate
./pyml.py \
    template "{model: ${MODEL}, SvB: ${SvB}}" $WFS/evaluate.yml \
    -from $WFS/../common.yml \
    -setting Monitor "address: :${port}"
