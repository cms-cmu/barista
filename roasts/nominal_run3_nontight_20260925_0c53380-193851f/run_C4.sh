#!/usr/bin/env bash
# roast nominal_run3_nontight_20260925_0c53380-193851f step C4 -- generated 2026-09-26 02:19:26
# Everything lives in main() so bash parses the whole file before executing: a later
# regeneration of this script cannot derail a run that is already in progress.
main() {
cd "$HOME/nobackup/HH4b/prod/nominal_run3_nontight_20260925_0c53380-193851f/barista" || exit 97
LOG=logs/C4.log
EXIT=logs/C4.exit
rm -f "$EXIT"
echo "=== roast nominal_run3_nontight_20260925_0c53380-193851f step C4 start $(date) on $(hostname) ===" | tee -a "$LOG"
echo "=== barista $(git rev-parse --short HEAD) coffea4bees $(cd coffea4bees && git rev-parse --short HEAD) ===" | tee -a "$LOG"
# Grid proxy: run_container binds ./proxy/x509_proxy into the container; seed it from the
# user's standard proxy (voms-proxy-init writes /tmp/x509up_u<uid>) when the checkout has none.
mkdir -p proxy
if [ ! -s proxy/x509_proxy ] || [ "${X509_USER_PROXY:-/tmp/x509up_u$(id -u)}" -nt proxy/x509_proxy ]; then
    cp -f "${X509_USER_PROXY:-/tmp/x509up_u$(id -u)}" proxy/x509_proxy 2>/dev/null && echo "=== proxy copied from ${X509_USER_PROXY:-/tmp/x509up_u$(id -u)} ===" | tee -a "$LOG"
fi
command -v voms-proxy-info >/dev/null && voms-proxy-info --file proxy/x509_proxy --timeleft 2>/dev/null | sed 's/^/=== proxy seconds left: /' | tee -a "$LOG"

./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseC_4_FvT_closure.smk --configfile roasts/nominal_run3_nontight_20260925_0c53380-193851f/config.yml --cores 8 --jobs 8 --printshellcmds --config roast_id=nominal_run3_nontight_20260925_0c53380-193851f 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "=== roast nominal_run3_nontight_20260925_0c53380-193851f step C4 exit $RC $(date) ===" | tee -a "$LOG"
echo "$RC" > "$EXIT"
if [ "$RC" != "0" ]; then
    echo "step C4 FAILED (rc=$RC). Shell kept open for inspection; exit to close."
    exec bash
fi
sleep 5
}
main "$@"
