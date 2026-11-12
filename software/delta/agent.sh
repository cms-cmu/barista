#!/usr/bin/env bash
# Start the Delta pull agent (software/delta/agent.py) from this barista checkout, in the pixi
# environment (xrdfs/xrdcp). Run it inside tmux on a Delta login node; see docs/software/delta.md.
#
#   tmux new -s delta-agent 'software/delta/agent.sh'      # options are passed to agent.py
#
# Restarts the agent if it exits (its state survives in $DELTA_HOME/agent_state.json).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
. software/delta/env.sh
export PATH="/projects/${DELTA_PROJECT}/${USER}/.pixi/bin:${HOME}/.pixi/bin:${PATH}"
command -v pixi >/dev/null || { echo "[delta-agent] no pixi: run './run_container snakemake --version' once to install it" >&2; exit 1; }
export X509_USER_PROXY="${X509_USER_PROXY:-${HOME}/.roast/x509_proxy}"
[ -d "${DELTA_SOFTWARE}/grid-security/certificates" ] && export X509_CERT_DIR="${DELTA_SOFTWARE}/grid-security/certificates"
while true; do
    pixi run -q python software/delta/agent.py "$@"
    rc=$?
    [ "$rc" -eq 0 ] && exit 0           # --once
    echo "[delta-agent] exited with ${rc}; restarting in 60 s (Ctrl-C to stop)" >&2
    sleep 60
done
