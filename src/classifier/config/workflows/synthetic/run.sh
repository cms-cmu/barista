SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# check port
if [ -z "$2" ]; then
    port=10200
else
    port=$2
fi

# train
for synthetic in {0..2}; do
    ./pyml.py template "{user: ${1}, synthetic: ${synthetic}}" $SCRIPT_DIR/train.yml \
    -setting Monitor "address: :${port}" -flag debug
done

# evaluate
for synthetic in {0..2}; do
    ./pyml.py template "{user: ${1}, synthetic: ${synthetic}}" $SCRIPT_DIR/evaluate.yml \
    -setting Monitor "address: :${port}"
done