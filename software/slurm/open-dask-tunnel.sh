#!/bin/bash
# Reverse SSH tunnel: exposes Dask dashboard (localhost:10200) on falcon.phys.cmu.edu:PORT
# Usage: export DASK_DASHBOARD_PORT=18800 && open-dask-tunnel.sh
set -euo pipefail

TUNNEL_KEY=/etc/dask-tunnel/id_ed25519
TUNNEL_USER=dask-tunnel
TUNNEL_HOST=falcon.phys.cmu.edu
PORT_MIN=10200
PORT_MAX=10300

if [[ -z "${DASK_DASHBOARD_PORT:-}" ]]; then
  echo "ERROR: DASK_DASHBOARD_PORT is not set. Export a port in ${PORT_MIN}-${PORT_MAX}." >&2
  exit 1
fi

if (( DASK_DASHBOARD_PORT < PORT_MIN || DASK_DASHBOARD_PORT > PORT_MAX )); then
  echo "ERROR: DASK_DASHBOARD_PORT=${DASK_DASHBOARD_PORT} is out of range ${PORT_MIN}-${PORT_MAX}." >&2
  exit 1
fi

SSH_OPTS="-i ${TUNNEL_KEY} -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=yes"

# Check if port is already bound on falcon
if ssh ${SSH_OPTS} "${TUNNEL_USER}@${TUNNEL_HOST}" \
    "ss -tlnp 2>/dev/null | grep -q ':${DASK_DASHBOARD_PORT} '" 2>/dev/null; then
  echo "ERROR: Port ${DASK_DASHBOARD_PORT} is already in use on ${TUNNEL_HOST}. Choose a different port." >&2
  exit 1
fi

# Open reverse tunnel in background
ssh -f -N \
    ${SSH_OPTS} \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -R "0.0.0.0:${DASK_DASHBOARD_PORT}:localhost:10200" \
    "${TUNNEL_USER}@${TUNNEL_HOST}"

echo "Dask dashboard: http://${TUNNEL_HOST}:${DASK_DASHBOARD_PORT}"
