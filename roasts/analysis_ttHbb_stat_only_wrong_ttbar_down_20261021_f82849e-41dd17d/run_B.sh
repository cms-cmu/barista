#!/usr/bin/env bash
# roast analysis_ttHbb_stat_only_wrong_ttbar_down_20261021_f82849e-41dd17d step B -- generated 2026-10-21 23:16:31
# Everything lives in main() so bash parses the whole file before executing: a later
# regeneration of this script cannot derail a run that is already in progress.
main() {
cd /uscms_data/d3/algomez/HH4b/prod/analysis_ttHbb_stat_only_wrong_ttbar_down_20261021_f82849e-41dd17d/barista || exit 97
LOG=logs/B.log
EXIT=logs/B.exit
rm -f "$EXIT"
echo "=== roast analysis_ttHbb_stat_only_wrong_ttbar_down_20261021_f82849e-41dd17d step B start $(date) on $(hostname) ===" | tee -a "$LOG"
echo "=== barista $(git rev-parse --short HEAD) coffea4bees $(cd coffea4bees && git rev-parse --short HEAD) ===" | tee -a "$LOG"
# Grid proxy: run_container binds ./proxy/x509_proxy into the container; seed it from the
# user's standard proxy (voms-proxy-init writes /tmp/x509up_u<uid>) or the shared copy
# `roast proxy` leaves in ~/.roast (LPC /tmp is node-local), whichever is newer than the
# checkout's. run_container `touch`es a missing proxy, so an empty file counts as none.
mkdir -p proxy
for _p in "${X509_USER_PROXY:-}" "$HOME/.roast/x509_proxy" /uscms_data/d3/algomez/HH4b/ref/barista/proxy/x509_proxy "/tmp/x509up_u$(id -u)"; do
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
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseB.smk --configfile roasts/analysis_ttHbb_stat_only_wrong_ttbar_down_20261021_f82849e-41dd17d/config.yml --cores 4 --jobs 4 --printshellcmds --config roast_id=analysis_ttHbb_stat_only_wrong_ttbar_down_20261021_f82849e-41dd17d eos_prod=root://cmseos.fnal.gov//store/user/algomez/HH4b_prod web_prod=root://eosuser.cern.ch//eos/user/a/algomez/www/ttHbb/prod --unlock || true
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseB.smk --configfile roasts/analysis_ttHbb_stat_only_wrong_ttbar_down_20261021_f82849e-41dd17d/config.yml --cores 4 --jobs 4 --printshellcmds --config roast_id=analysis_ttHbb_stat_only_wrong_ttbar_down_20261021_f82849e-41dd17d eos_prod=root://cmseos.fnal.gov//store/user/algomez/HH4b_prod web_prod=root://eosuser.cern.ch//eos/user/a/algomez/www/ttHbb/prod --rerun-incomplete 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "=== roast analysis_ttHbb_stat_only_wrong_ttbar_down_20261021_f82849e-41dd17d step B exit $RC $(date) ===" | tee -a "$LOG"
echo "$RC" > "$EXIT"
if [ "$RC" != "0" ]; then
    echo "step B FAILED (rc=$RC). Shell kept open for inspection; exit to close."
    exec bash
fi
sleep 5
}
main "$@"
