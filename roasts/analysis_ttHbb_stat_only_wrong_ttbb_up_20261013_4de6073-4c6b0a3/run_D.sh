#!/usr/bin/env bash
# roast analysis_ttHbb_stat_only_wrong_ttbb_up_20261013_4de6073-4c6b0a3 step D -- generated 2026-10-14 23:30:27
# Everything lives in main() so bash parses the whole file before executing: a later
# regeneration of this script cannot derail a run that is already in progress.
main() {
cd /home/export/tgomezes/workingArea/roast_prod/analysis_ttHbb_stat_only_wrong_ttbb_up_20261013_4de6073-4c6b0a3/barista || exit 97
LOG=logs/D.log
EXIT=logs/D.exit
rm -f "$EXIT"
echo "=== roast analysis_ttHbb_stat_only_wrong_ttbb_up_20261013_4de6073-4c6b0a3 step D start $(date) on $(hostname) ===" | tee -a "$LOG"
echo "=== barista $(git rev-parse --short HEAD) coffea4bees $(cd coffea4bees && git rev-parse --short HEAD) ===" | tee -a "$LOG"
# Grid proxy: run_container binds ./proxy/x509_proxy into the container; seed it from the
# user's standard proxy (voms-proxy-init writes /tmp/x509up_u<uid>) when the checkout has none.
mkdir -p proxy
if [ ! -s proxy/x509_proxy ] || [ "${X509_USER_PROXY:-/tmp/x509up_u$(id -u)}" -nt proxy/x509_proxy ]; then
    cp -f "${X509_USER_PROXY:-/tmp/x509up_u$(id -u)}" proxy/x509_proxy 2>/dev/null && echo "=== proxy copied from ${X509_USER_PROXY:-/tmp/x509up_u$(id -u)} ===" | tee -a "$LOG"
fi
if [ -s proxy/kubeconfig ]; then
    export KUBECONFIG="${PWD}/proxy/kubeconfig"
elif [ -s "$HOME/.kube/config" ]; then
    export KUBECONFIG="$HOME/.kube/config"
fi
command -v voms-proxy-info >/dev/null && voms-proxy-info --file proxy/x509_proxy --timeleft 2>/dev/null | sed 's/^/=== proxy seconds left: /' | tee -a "$LOG"

./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseD.smk --configfile roasts/analysis_ttHbb_stat_only_wrong_ttbb_up_20261013_4de6073-4c6b0a3/config.yml --cores 4 --jobs 4 --printshellcmds --config roast_id=analysis_ttHbb_stat_only_wrong_ttbb_up_20261013_4de6073-4c6b0a3 eos_prod=root://cmseos.fnal.gov//store/user/algomez/HH4b_prod web_prod=root://eosuser.cern.ch//eos/user/a/algomez/www/ttHbb/prod 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "=== roast analysis_ttHbb_stat_only_wrong_ttbb_up_20261013_4de6073-4c6b0a3 step D exit $RC $(date) ===" | tee -a "$LOG"
echo "$RC" > "$EXIT"
if [ "$RC" != "0" ]; then
    echo "step D FAILED (rc=$RC). Shell kept open for inspection; exit to close."
    exec bash
fi
sleep 5
}
main "$@"
