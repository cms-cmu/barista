#!/usr/bin/env bash
# roast svbfeynnet_run3_20260923_5b01004-31a4b4d step SvBFeynNet_friendtrees_Run3 -- generated 2026-09-23 16:14:32
# Everything lives in main() so bash parses the whole file before executing: a later
# regeneration of this script cannot derail a run that is already in progress.
main() {
cd "$HOME/nobackup/HH4b/prod/svbfeynnet_run3_20260923_5b01004-31a4b4d/barista" || exit 97
LOG=logs/SvBFeynNet_friendtrees_Run3.log
EXIT=logs/SvBFeynNet_friendtrees_Run3.exit
rm -f "$EXIT"
echo "=== roast svbfeynnet_run3_20260923_5b01004-31a4b4d step SvBFeynNet_friendtrees_Run3 start $(date) on $(hostname) ===" | tee -a "$LOG"
echo "=== barista $(git rev-parse --short HEAD) coffea4bees $(cd coffea4bees && git rev-parse --short HEAD) ===" | tee -a "$LOG"
# Grid proxy: run_container binds ./proxy/x509_proxy into the container; seed it from the
# user's standard proxy (voms-proxy-init writes /tmp/x509up_u<uid>) when the checkout has none.
mkdir -p proxy
if [ ! -s proxy/x509_proxy ] || [ "${X509_USER_PROXY:-/tmp/x509up_u$(id -u)}" -nt proxy/x509_proxy ]; then
    cp -f "${X509_USER_PROXY:-/tmp/x509up_u$(id -u)}" proxy/x509_proxy 2>/dev/null && echo "=== proxy copied from ${X509_USER_PROXY:-/tmp/x509up_u$(id -u)} ===" | tee -a "$LOG"
fi
command -v voms-proxy-info >/dev/null && voms-proxy-info --file proxy/x509_proxy --timeleft 2>/dev/null | sed 's/^/=== proxy seconds left: /' | tee -a "$LOG"

./run_container snakemake -s coffea4bees/workflows/Snakefile_SvBFeynNet_friendtrees_Run3.smk --configfile roasts/svbfeynnet_run3_20260923_5b01004-31a4b4d/config.yml --cores 8 --jobs 8 --printshellcmds --config roast_id=svbfeynnet_run3_20260923_5b01004-31a4b4d --forcerun install_SvBFeynNet_friend_json --config test=true 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "=== roast svbfeynnet_run3_20260923_5b01004-31a4b4d step SvBFeynNet_friendtrees_Run3 exit $RC $(date) ===" | tee -a "$LOG"
echo "$RC" > "$EXIT"
if [ "$RC" != "0" ]; then
    echo "step SvBFeynNet_friendtrees_Run3 FAILED (rc=$RC). Shell kept open for inspection; exit to close."
    exec bash
fi
sleep 5
}
main "$@"
