# NCSA Delta: resolve the ACCESS project whose accounts and storage barista uses.
# Sourced by run_container and software/bin/apptainer; see docs/delta.md.
#
# Precedence: environment > ${XDG_CONFIG_HOME:-~/.config}/barista/delta.env > bjsv (the original
# project, so an unconfigured account behaves exactly as before). Example delta.env:
#     DELTA_PROJECT=bjsy
#     DELTA_CPU_ACCOUNT=bjsy-delta-cpu   # only if it differs from <project>-delta-cpu
_delta_env_project=${DELTA_PROJECT-}
_delta_env_gpu=${DELTA_GPU_ACCOUNT-}
_delta_env_cpu=${DELTA_CPU_ACCOUNT-}
_delta_env_sw=${DELTA_SOFTWARE-}
_delta_env_file="${XDG_CONFIG_HOME:-$HOME/.config}/barista/delta.env"
# shellcheck disable=SC1090
[ -r "$_delta_env_file" ] && . "$_delta_env_file"
DELTA_PROJECT=${_delta_env_project:-${DELTA_PROJECT:-bjsv}}
DELTA_GPU_ACCOUNT=${_delta_env_gpu:-${DELTA_GPU_ACCOUNT:-${DELTA_PROJECT}-delta-gpu}}
DELTA_CPU_ACCOUNT=${_delta_env_cpu:-${DELTA_CPU_ACCOUNT:-${DELTA_PROJECT}-delta-cpu}}
# cvmfsexec and grid certificates; point at another project's copy if it is readable to you
DELTA_SOFTWARE=${_delta_env_sw:-${DELTA_SOFTWARE:-/projects/${DELTA_PROJECT}/software}}
export DELTA_PROJECT DELTA_GPU_ACCOUNT DELTA_CPU_ACCOUNT DELTA_SOFTWARE
unset _delta_env_project _delta_env_gpu _delta_env_cpu _delta_env_sw _delta_env_file
