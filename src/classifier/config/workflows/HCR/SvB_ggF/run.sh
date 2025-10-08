export LPCUSER="chuyuanl"
export CERNUSER="c/chuyuan"
export WFS="classifier/config/workflows/HCR/SvB_ggF"
export BASE="root://cmseos.fnal.gov//store/user/${LPCUSER}/HH4b"
export MODEL="${BASE}/classifier/SvB/ggF/$1/result.json"
export WEB="root://eosuser.cern.ch//eos/user/${CERNUSER}/www/HH4b/classifier/SvB_ggF/"
export GMAIL=~/gmail.yml

# check port
if [ -z "$2" ]; then
    port=10200
else
    port=$2
fi

# background normalization
declare -A norms
norms[${1}]=3
norms["all_no_ZX"]=4
norms["sm_no_ZX"]=1
norms["kl5_no_ZX"]=1
norms["all_kl"]=6
norms["ZX_only"]=2

# additional setting
declare -a other_setting=()
if [[ $1 == *"no_ZX"* ]]; then
    other_setting+=('-setting' 'ml.MultiClass' 'nontrainable_labels: [ZZ, ZH]')
fi

if [[ $1 == "ZX_only" ]]; then
    other_setting+=('-setting' 'ml.MultiClass' 'nontrainable_labels: [ggF]')
fi

# train and make plots
./pyml.py \
    template "{norm: ${norms[$1]}, user: ${LPCUSER}, model: $1}" $WFS/train.yml \
    -setting Monitor "address: :${port}" -flag debug "${other_setting[@]}"

./pyml.py analyze --results ${MODEL} \
    -analysis HCR.LossROC \
    -setting IO "output: ${WEB}" \
    -setting IO "report: $1" \
    -setting Monitor "address: :${port}"
# evaluate
./pyml.py \
    template "{user: ${LPCUSER}, model: $1}" $WFS/evaluate.yml \
    -setting Monitor "address: :${port}"

if [ -e "$GMAIL" ]; then
    ./pyml.py analyze \
        -analysis notify.Gmail \
        --title "SvB jobs done" \
        --body "${1} finished at $(date)" \
        --labels Classifier HH4b \
        -from $GMAIL \
        -setting Monitor "address: :${port}"
fi