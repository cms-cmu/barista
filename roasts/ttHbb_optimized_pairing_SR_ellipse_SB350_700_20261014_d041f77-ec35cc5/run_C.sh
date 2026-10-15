#!/usr/bin/env bash
# roast ttHbb_optimized_pairing_SR_ellipse_SB350_700_20261014_d041f77-ec35cc5 step C -- generated 2026-10-14 18:08:43
# Everything lives in main() so bash parses the whole file before executing: a later
# regeneration of this script cannot derail a run that is already in progress.
main() {
cd /home/export/sindhum/roast_prod/ttHbb_optimized_pairing_SR_ellipse_SB350_700_20261014_d041f77-ec35cc5/barista || exit 97
LOG=logs/C.log
EXIT=logs/C.exit
rm -f "$EXIT"
echo "=== roast ttHbb_optimized_pairing_SR_ellipse_SB350_700_20261014_d041f77-ec35cc5 step C start $(date) on $(hostname) ===" | tee -a "$LOG"
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

./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseC.smk --configfile roasts/ttHbb_optimized_pairing_SR_ellipse_SB350_700_20261014_d041f77-ec35cc5/config.yml --cores 4 --jobs 4 --printshellcmds --config roast_id=ttHbb_optimized_pairing_SR_ellipse_SB350_700_20261014_d041f77-ec35cc5 eos_prod=root://cmseos.fnal.gov//store/user/smurthy/ttHbb web_prod=root://eosuser.cern.ch//eos/user/s/simurthy/www/ttHbb 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "=== roast ttHbb_optimized_pairing_SR_ellipse_SB350_700_20261014_d041f77-ec35cc5 step C exit $RC $(date) ===" | tee -a "$LOG"
echo "$RC" > "$EXIT"
if [ "$RC" != "0" ]; then
    echo "step C FAILED (rc=$RC). Shell kept open for inspection; exit to close."
    exec bash
fi
sleep 5
}
main "$@"
