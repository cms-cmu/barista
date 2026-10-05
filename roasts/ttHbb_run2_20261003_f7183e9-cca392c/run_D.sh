#!/usr/bin/env bash
# roast ttHbb_run2_20261003_f7183e9-cca392c step D -- generated 2026-10-04 04:10:56
# Everything lives in main() so bash parses the whole file before executing: a later
# regeneration of this script cannot derail a run that is already in progress.
main() {
cd /home/export/tgomezes/workingArea/roast_prod/ttHbb_run2_20261003_f7183e9-cca392c/barista || exit 97
LOG=logs/D.log
EXIT=logs/D.exit
rm -f "$EXIT"
echo "=== roast ttHbb_run2_20261003_f7183e9-cca392c step D start $(date) on $(hostname) ===" | tee -a "$LOG"
echo "=== barista $(git rev-parse --short HEAD) coffea4bees $(cd coffea4bees && git rev-parse --short HEAD) ===" | tee -a "$LOG"
# Grid proxy: run_container binds ./proxy/x509_proxy into the container; seed it from the
# user's standard proxy (voms-proxy-init writes /tmp/x509up_u<uid>) when the checkout has none.
mkdir -p proxy
if [ ! -s proxy/x509_proxy ] || [ "${X509_USER_PROXY:-/tmp/x509up_u$(id -u)}" -nt proxy/x509_proxy ]; then
    cp -f "${X509_USER_PROXY:-/tmp/x509up_u$(id -u)}" proxy/x509_proxy 2>/dev/null && echo "=== proxy copied from ${X509_USER_PROXY:-/tmp/x509up_u$(id -u)} ===" | tee -a "$LOG"
fi
command -v voms-proxy-info >/dev/null && voms-proxy-info --file proxy/x509_proxy --timeleft 2>/dev/null | sed 's/^/=== proxy seconds left: /' | tee -a "$LOG"

./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseD.smk --configfile roasts/ttHbb_run2_20261003_f7183e9-cca392c/config.yml --cores 4 --jobs 4 --printshellcmds --config roast_id=ttHbb_run2_20261003_f7183e9-cca392c eos_prod=root://cmseos.fnal.gov//store/user/algomez/HH4b_prod web_prod=root://eosuser.cern.ch//eos/user/a/algomez/www/HH4b/prod 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "=== roast ttHbb_run2_20261003_f7183e9-cca392c step D exit $RC $(date) ===" | tee -a "$LOG"
echo "$RC" > "$EXIT"
if [ "$RC" != "0" ]; then
    echo "step D FAILED (rc=$RC). Shell kept open for inspection; exit to close."
    exec bash
fi
sleep 5
}
main "$@"
