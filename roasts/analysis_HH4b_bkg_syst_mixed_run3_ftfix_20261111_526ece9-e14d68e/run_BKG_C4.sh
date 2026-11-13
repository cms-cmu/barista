#!/usr/bin/env bash
# roast analysis_HH4b_bkg_syst_mixed_run3_ftfix_20261111_526ece9-e14d68e step BKG_C4 -- generated 2026-11-12 17:20:45
# Everything lives in main() so bash parses the whole file before executing: a later
# regeneration of this script cannot derail a run that is already in progress.
main() {
cd "$HOME/nobackup/HH4b/prod/analysis_HH4b_bkg_syst_mixed_run3_ftfix_20261111_526ece9-e14d68e/barista" || exit 97
LOG=logs/BKG_C4.log
EXIT=logs/BKG_C4.exit
rm -f "$EXIT"
echo "=== roast analysis_HH4b_bkg_syst_mixed_run3_ftfix_20261111_526ece9-e14d68e step BKG_C4 start $(date) on $(hostname) ===" | tee -a "$LOG"
echo "=== barista $(git rev-parse --short HEAD) coffea4bees $(cd coffea4bees && git rev-parse --short HEAD) ===" | tee -a "$LOG"
# Grid proxy: run_container binds ./proxy/x509_proxy into the container; seed it from the
# user's standard proxy (voms-proxy-init writes /tmp/x509up_u<uid>) or the shared copy
# `roast proxy` leaves in ~/.roast (LPC /tmp is node-local), whichever is newer than the
# checkout's. run_container `touch`es a missing proxy, so an empty file counts as none.
mkdir -p proxy
for _p in "${X509_USER_PROXY:-}" "$HOME/.roast/x509_proxy" "$HOME/nobackup/HH4b/Run3/barista/proxy/x509_proxy" "/tmp/x509up_u$(id -u)"; do
    if [ -n "$_p" ] && [ -s "$_p" ] && voms-proxy-info -exists -valid 0:05 -file "$_p" >/dev/null 2>&1; then
        cp -f "$_p" proxy/x509_proxy && echo "=== proxy copied from $_p ===" | tee -a "$LOG"
        break
    fi
done
[ -s proxy/x509_proxy ] && export X509_USER_PROXY="${PWD}/proxy/x509_proxy"
if [ -s proxy/kubeconfig ]; then
    export KUBECONFIG="${PWD}/proxy/kubeconfig"
elif [ -s "$HOME/.kube/config" ]; then
    export KUBECONFIG="$HOME/.kube/config"
fi
command -v voms-proxy-info >/dev/null && voms-proxy-info --file proxy/x509_proxy --timeleft 2>/dev/null | sed 's/^/=== proxy seconds left: /' | tee -a "$LOG"

# 1. Capture visual rulegraph before execution
echo "=== generating rulegraph SVG ===" | tee -a "$LOG"
./run_container snakemake --rulegraph -s coffea4bees/workflows/Snakefile_bkg_syst_C_4_closure.smk --configfile roasts/analysis_HH4b_bkg_syst_mixed_run3_ftfix_20261111_526ece9-e14d68e/config.yml --config roast_id=analysis_HH4b_bkg_syst_mixed_run3_ftfix_20261111_526ece9-e14d68e eos_prod=root://cmseos.fnal.gov//store/user/jda102/HH4b_prod web_prod=root://eosuser.cern.ch//eos/user/j/johnda/www/HH4b/prod train_host=orchard eval_host=orchard 2>/dev/null | dot -Tsvg > "logs/BKG_C4_rulegraph.svg" 2>/dev/null || true
# 2. Run workflow
./run_container snakemake -s coffea4bees/workflows/Snakefile_bkg_syst_C_4_closure.smk --configfile roasts/analysis_HH4b_bkg_syst_mixed_run3_ftfix_20261111_526ece9-e14d68e/config.yml --cores 8 --jobs 8 --printshellcmds --config roast_id=analysis_HH4b_bkg_syst_mixed_run3_ftfix_20261111_526ece9-e14d68e eos_prod=root://cmseos.fnal.gov//store/user/jda102/HH4b_prod web_prod=root://eosuser.cern.ch//eos/user/j/johnda/www/HH4b/prod train_host=orchard eval_host=orchard 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "=== roast analysis_HH4b_bkg_syst_mixed_run3_ftfix_20261111_526ece9-e14d68e step BKG_C4 exit $RC $(date) ===" | tee -a "$LOG"
echo "$RC" > "$EXIT"
if [ "$RC" = "0" ]; then
    # 3. Capture snakemake execution summary on success
    echo "=== generating snakemake execution summary ===" | tee -a "$LOG"
    ./run_container snakemake --summary -s coffea4bees/workflows/Snakefile_bkg_syst_C_4_closure.smk --configfile roasts/analysis_HH4b_bkg_syst_mixed_run3_ftfix_20261111_526ece9-e14d68e/config.yml --config roast_id=analysis_HH4b_bkg_syst_mixed_run3_ftfix_20261111_526ece9-e14d68e eos_prod=root://cmseos.fnal.gov//store/user/jda102/HH4b_prod web_prod=root://eosuser.cern.ch//eos/user/j/johnda/www/HH4b/prod train_host=orchard eval_host=orchard > "logs/BKG_C4_summary.txt" 2>/dev/null || true
fi
if [ "$RC" != "0" ]; then
    echo "step BKG_C4 FAILED (rc=$RC). Shell kept open for inspection; exit to close."
    exec bash
fi
sleep 5
}
main "$@"
