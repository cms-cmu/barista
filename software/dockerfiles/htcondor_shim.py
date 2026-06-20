from htcondor2 import *
from htcondor2 import param, Schedd, Collector, Submit as _RealSubmit, JobEventLog, JobAction, DaemonType, AdTypes
_submit_results = {}
_real_submit = Schedd.submit
_real_spool = Schedd.spool
def patched_submit(self, submit_obj, *args, **kwargs):
    result = _real_submit(self, submit_obj, *args, **kwargs)
    try:
        cluster_id = result.cluster()
        _submit_results[cluster_id] = result
    except Exception: pass
    return result
def patched_spool(self, jobs_or_result):
    if isinstance(jobs_or_result, list) and len(jobs_or_result) > 0:
        first_job = jobs_or_result[0]
        cluster_id = None
        if hasattr(first_job, 'get'):
            cluster_id = first_job.get('ClusterId')
        elif isinstance(first_job, dict):
            cluster_id = first_job.get('ClusterId')
        if cluster_id is not None and cluster_id in _submit_results:
            result = _submit_results[cluster_id]
            return _real_spool(self, result)
        else:
            raise RuntimeError(f'Could not find SubmitResult for cluster ID {cluster_id}')
    else:
        return _real_spool(self, jobs_or_result)
Schedd.submit = patched_submit
Schedd.spool = patched_spool
class Submit(_RealSubmit):
    def jobs(self, clusterid=None):
        num_procs = 1
        try:
            if hasattr(self, 'procs'):
                num_procs = self.procs()
        except Exception: pass
        return [{'ClusterId': clusterid, 'ProcId': i} for i in range(num_procs)]
