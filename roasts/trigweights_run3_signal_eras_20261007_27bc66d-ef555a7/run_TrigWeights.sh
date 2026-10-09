#!/usr/bin/env bash
# roast trigweights_run3_signal_eras_20261007_27bc66d-ef555a7 step TrigWeights -- generated 2026-10-07 22:35:23
# Everything lives in main() so bash parses the whole file before executing: a later
# regeneration of this script cannot derail a run that is already in progress.
main() {
cd "$HOME/nobackup/HH4b/prod/trigweights_run3_signal_eras_20261007_27bc66d-ef555a7/barista" || exit 97
LOG=logs/TrigWeights.log
EXIT=logs/TrigWeights.exit
rm -f "$EXIT"
echo "=== roast trigweights_run3_signal_eras_20261007_27bc66d-ef555a7 step TrigWeights start $(date) on $(hostname) ===" | tee -a "$LOG"
echo "=== barista $(git rev-parse --short HEAD) coffea4bees $(cd coffea4bees && git rev-parse --short HEAD) ===" | tee -a "$LOG"
# Grid proxy: run_container binds ./proxy/x509_proxy into the container; seed it from the
# user's standard proxy (voms-proxy-init writes /tmp/x509up_u<uid>) when the checkout has none.
mkdir -p proxy
if [ ! -s proxy/x509_proxy ] || [ "${X509_USER_PROXY:-/tmp/x509up_u$(id -u)}" -nt proxy/x509_proxy ]; then
    cp -f "${X509_USER_PROXY:-/tmp/x509up_u$(id -u)}" proxy/x509_proxy 2>/dev/null && echo "=== proxy copied from ${X509_USER_PROXY:-/tmp/x509up_u$(id -u)} ===" | tee -a "$LOG"
fi
command -v voms-proxy-info >/dev/null && voms-proxy-info --file proxy/x509_proxy --timeleft 2>/dev/null | sed 's/^/=== proxy seconds left: /' | tee -a "$LOG"
./run_container snakemake -s coffea4bees/workflows/Snakefile_TrigWeights.smk --configfile roasts/trigweights_run3_signal_eras_20261007_27bc66d-ef555a7/config.yml --cores 2 --jobs 2 --printshellcmds --config roast_id=trigweights_run3_signal_eras_20261007_27bc66d-ef555a7 eos_prod=root://cmseos.fnal.gov//store/user/jda102/HH4b_prod web_prod=root://eosuser.cern.ch//eos/user/j/johnda/www/HH4b/prod --unlock || true
./run_container snakemake -s coffea4bees/workflows/Snakefile_TrigWeights.smk --configfile roasts/trigweights_run3_signal_eras_20261007_27bc66d-ef555a7/config.yml --cores 2 --jobs 2 --printshellcmds --config roast_id=trigweights_run3_signal_eras_20261007_27bc66d-ef555a7 eos_prod=root://cmseos.fnal.gov//store/user/jda102/HH4b_prod web_prod=root://eosuser.cern.ch//eos/user/j/johnda/www/HH4b/prod --rerun-incomplete 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "=== roast trigweights_run3_signal_eras_20261007_27bc66d-ef555a7 step TrigWeights exit $RC $(date) ===" | tee -a "$LOG"
echo "$RC" > "$EXIT"
if [ "$RC" != "0" ]; then
    echo "step TrigWeights FAILED (rc=$RC). Shell kept open for inspection; exit to close."
    exec bash
fi
sleep 5
}
main "$@"
