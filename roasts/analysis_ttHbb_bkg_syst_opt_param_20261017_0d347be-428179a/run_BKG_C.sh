#!/usr/bin/env bash
# roast analysis_ttHbb_bkg_syst_opt_param_20261017_0d347be-428179a step BKG_C -- generated 2026-10-17 13:52:30
# Everything lives in main() so bash parses the whole file before executing: a later
# regeneration of this script cannot derail a run that is already in progress.
main() {
cd /ocean/projects/phy260026p/tgomezes/HH4b/prod/analysis_ttHbb_bkg_syst_opt_param_20261017_0d347be-428179a/barista || exit 97
LOG=logs/BKG_C.log
EXIT=logs/BKG_C.exit
rm -f "$EXIT"
echo "=== roast analysis_ttHbb_bkg_syst_opt_param_20261017_0d347be-428179a step BKG_C start $(date) on $(hostname) ===" | tee -a "$LOG"
echo "=== barista $(git rev-parse --short HEAD) coffea4bees $(cd coffea4bees && git rev-parse --short HEAD) ===" | tee -a "$LOG"
# Grid proxy: run_container binds ./proxy/x509_proxy into the container; seed it from the
# user's standard proxy (voms-proxy-init writes /tmp/x509up_u<uid>) when the checkout has none.
mkdir -p proxy
for _p in "${X509_USER_PROXY:-}" "$HOME/.roast/x509_proxy" /ocean/projects/phy260026p/tgomezes/HH4b/barista/proxy/x509_proxy "/tmp/x509up_u$(id -u)"; do
    if [ -n "$_p" ] && [ -s "$_p" ] && voms-proxy-info -exists -valid 0:05 -file "$_p" >/dev/null 2>&1; then
        cp -f "$_p" proxy/x509_proxy && echo "=== proxy copied from $_p ===" | tee -a "$LOG"
        break
    fi
done
if [ -s proxy/kubeconfig ]; then
    export KUBECONFIG="${PWD}/proxy/kubeconfig"
elif [ -s "$HOME/.kube/config" ]; then
    export KUBECONFIG="$HOME/.kube/config"
fi
command -v voms-proxy-info >/dev/null && voms-proxy-info --file proxy/x509_proxy --timeleft 2>/dev/null | sed 's/^/=== proxy seconds left: /' | tee -a "$LOG"

./run_container snakemake -s coffea4bees/workflows/Snakefile_bkg_syst_C.smk all_bkg_syst_C --configfile roasts/analysis_ttHbb_bkg_syst_opt_param_20261017_0d347be-428179a/config.yml --cores 16 --jobs 16 --printshellcmds --config roast_id=analysis_ttHbb_bkg_syst_opt_param_20261017_0d347be-428179a eos_prod=root://cmseos.fnal.gov//store/user/algomez/HH4b_prod web_prod=root://eosuser.cern.ch//eos/user/a/algomez/www/ttHbb/prod 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "=== roast analysis_ttHbb_bkg_syst_opt_param_20261017_0d347be-428179a step BKG_C exit $RC $(date) ===" | tee -a "$LOG"
echo "$RC" > "$EXIT"
if [ "$RC" != "0" ]; then
    echo "step BKG_C FAILED (rc=$RC). Shell kept open for inspection; exit to close."
    exec bash
fi
sleep 5
}
main "$@"
