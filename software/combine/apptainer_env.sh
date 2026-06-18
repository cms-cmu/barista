# Dynamic CMSSW environment setup script (to be sourced via BASH_ENV / ENV)

if [ -z "${BARISTA_ENV_SOURCED:-}" ]; then
    export BARISTA_ENV_SOURCED=1
    
    CMSSW_DIR="/home/cmsusr/CMSSW_14_1_0_pre4"
    if [ -n "$CMSSW_DIR" ] && [ -d "$CMSSW_DIR" ]; then
        source /cvmfs/cms.cern.ch/cmsset_default.sh
        cd "$CMSSW_DIR"
        eval $(scram runtime -sh)
        cd - > /dev/null
    fi
fi
