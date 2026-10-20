#!/usr/bin/env bash
# roast HH4b_bkg_syst_declib16_run3_20261017_75a0c1f-0c8bb0c step BKG_AB -- generated 2026-10-17 17:18:43
# Everything lives in main() so bash parses the whole file before executing: a later
# regeneration of this script cannot derail a run that is already in progress.
main() {
cd "$HOME/nobackup/HH4b/prod/HH4b_bkg_syst_declib16_run3_20261017_75a0c1f-0c8bb0c/barista" || exit 97
LOG=logs/BKG_AB.log
EXIT=logs/BKG_AB.exit
rm -f "$EXIT"
echo "=== roast HH4b_bkg_syst_declib16_run3_20261017_75a0c1f-0c8bb0c step BKG_AB start $(date) on $(hostname) ===" | tee -a "$LOG"
echo "=== barista $(git rev-parse --short HEAD) coffea4bees $(cd coffea4bees && git rev-parse --short HEAD) ===" | tee -a "$LOG"
# Grid proxy: run_container binds ./proxy/x509_proxy into the container; seed it from the
# user's standard proxy (voms-proxy-init writes /tmp/x509up_u<uid>) or the shared copy
# `roast proxy` leaves in ~/.roast (LPC /tmp is node-local), whichever is newer than the
# checkout's. run_container `touch`es a missing proxy, so an empty file counts as none.
mkdir -p proxy
for src in "${X509_USER_PROXY:-/tmp/x509up_u$(id -u)}" "$HOME/.roast/x509_proxy"; do
    if [ -s "$src" ] && { [ ! -s proxy/x509_proxy ] || [ "$src" -nt proxy/x509_proxy ]; }; then
        cp -f "$src" proxy/x509_proxy && echo "=== proxy copied from $src ===" | tee -a "$LOG"
    fi
done
if [ -s proxy/kubeconfig ]; then
    export KUBECONFIG="${PWD}/proxy/kubeconfig"
elif [ -s "$HOME/.kube/config" ]; then
    export KUBECONFIG="$HOME/.kube/config"
fi
command -v voms-proxy-info >/dev/null && voms-proxy-info --file proxy/x509_proxy --timeleft 2>/dev/null | sed 's/^/=== proxy seconds left: /' | tee -a "$LOG"

./run_container snakemake -s coffea4bees/workflows/Snakefile_bkg_syst_AB.smk --configfile roasts/HH4b_bkg_syst_declib16_run3_20261017_75a0c1f-0c8bb0c/config.yml --cores 8 --jobs 8 --printshellcmds --config roast_id=HH4b_bkg_syst_declib16_run3_20261017_75a0c1f-0c8bb0c eos_prod=root://cmseos.fnal.gov//store/user/jda102/HH4b_prod web_prod=root://eosuser.cern.ch//eos/user/j/johnda/www/HH4b/prod 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "=== roast HH4b_bkg_syst_declib16_run3_20261017_75a0c1f-0c8bb0c step BKG_AB exit $RC $(date) ===" | tee -a "$LOG"
echo "$RC" > "$EXIT"
if [ "$RC" != "0" ]; then
    echo "step BKG_AB FAILED (rc=$RC). Shell kept open for inspection; exit to close."
    exec bash
fi
sleep 5
}
main "$@"
