export LPCUSER="chuyuanl"
export CERNUSER="c/chuyuan"
export WFS="classifier/config/workflows/HCR/SvB_ggF_arch"
export BASE="root://cmseos.fnal.gov//store/user/${LPCUSER}/HH4b"
export MODEL="${BASE}/classifier/SvB/ggF_arch/${1}/result.json"
export WEB="root://eosuser.cern.ch//eos/user/${CERNUSER}/www/HH4b/classifier/SvB_ggF_arch/"
export GMAIL=~/gmail.yml

# check port
if [ -z "$2" ]; then
    port=10200
else
    port=$2
fi


# train and make plots
./pyml.py \
    template "{user: ${LPCUSER}, arch: ${1}}" $WFS/train.yml \
    -setting Monitor "address: :${port}"

./pyml.py analyze --results ${MODEL} \
    -analysis HCR.LossROC \
    -setting IO "output: ${WEB}" \
    -setting IO "report: $1" \
    -setting Monitor "address: :${port}"
