#!/usr/bin/env bash
# roast mixeddata_run3_20260926_956d4bf-aca8a4e step MakeMixedData -- generated 2026-09-27 23:19:00
# Everything lives in main() so bash parses the whole file before executing: a later
# regeneration of this script cannot derail a run that is already in progress.
main() {
cd "$HOME/nobackup/HH4b/prod/mixeddata_run3_20260926_956d4bf-aca8a4e/barista" || exit 97
LOG=logs/MakeMixedData.log
EXIT=logs/MakeMixedData.exit
rm -f "$EXIT"
echo "=== roast mixeddata_run3_20260926_956d4bf-aca8a4e step MakeMixedData start $(date) on $(hostname) ===" | tee -a "$LOG"
echo "=== barista $(git rev-parse --short HEAD) coffea4bees $(cd coffea4bees && git rev-parse --short HEAD) ===" | tee -a "$LOG"
# Grid proxy: run_container binds ./proxy/x509_proxy into the container; seed it from the
# user's standard proxy (voms-proxy-init writes /tmp/x509up_u<uid>) when the checkout has none.
mkdir -p proxy
if [ ! -s proxy/x509_proxy ] || [ "${X509_USER_PROXY:-/tmp/x509up_u$(id -u)}" -nt proxy/x509_proxy ]; then
    cp -f "${X509_USER_PROXY:-/tmp/x509up_u$(id -u)}" proxy/x509_proxy 2>/dev/null && echo "=== proxy copied from ${X509_USER_PROXY:-/tmp/x509up_u$(id -u)} ===" | tee -a "$LOG"
fi
command -v voms-proxy-info >/dev/null && voms-proxy-info --file proxy/x509_proxy --timeleft 2>/dev/null | sed 's/^/=== proxy seconds left: /' | tee -a "$LOG"

./run_container snakemake -s coffea4bees/workflows/Snakefile_MakeMixedData.smk all_M4 all_M6 --configfile roasts/mixeddata_run3_20260926_956d4bf-aca8a4e/config.yml --cores 8 --jobs 8 --printshellcmds --config roast_id=mixeddata_run3_20260926_956d4bf-aca8a4e --jobs 3 --forcerun output/mixeddata_run3_20260926_956d4bf-aca8a4e/M4/per_subsample/picoaod_datasets_mixeddata_all_v0.yml output/mixeddata_run3_20260926_956d4bf-aca8a4e/M4/per_subsample/picoaod_datasets_mixeddata_all_v3.yml output/mixeddata_run3_20260926_956d4bf-aca8a4e/M4/per_subsample/picoaod_datasets_mixeddata_all_v4.yml output/mixeddata_run3_20260926_956d4bf-aca8a4e/M4/per_subsample/picoaod_datasets_mixeddata_all_v12.yml output/mixeddata_run3_20260926_956d4bf-aca8a4e/M4/per_subsample/picoaod_datasets_mixeddata_all_v15.yml 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "=== roast mixeddata_run3_20260926_956d4bf-aca8a4e step MakeMixedData exit $RC $(date) ===" | tee -a "$LOG"
echo "$RC" > "$EXIT"
if [ "$RC" != "0" ]; then
    echo "step MakeMixedData FAILED (rc=$RC). Shell kept open for inspection; exit to close."
    exec bash
fi
sleep 5
}
main "$@"
