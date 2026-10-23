"""Diagnostic CUDA launch provenance; never a timing or coverage exemption."""
from collections import defaultdict


def launch_provenance(cpu, gpu):
    """Correlate native CUDA events to CPU launches and enclosing annotations.

    GPU timestamps need not lie inside the CPU scope (execution is async).
    Retain ambiguous/missing correlations rather than guessing an owner.
    """
    launches = defaultdict(list)
    for event in cpu:
        if event.activity_type() in ('cuda_runtime', 'cuda_driver'):
            launches[event.correlation_id()].append(event)
    scopes = [event for event in cpu if event.is_user_annotation()
              or event.activity_type() not in ('cuda_runtime', 'cuda_driver')]
    rows = []
    for event in gpu:
        if event.is_user_annotation():
            continue
        sources = []
        for launch in launches.get(event.correlation_id(), []):
            enclosing = [scope for scope in scopes
                         if scope.device_resource_id() == launch.device_resource_id()
                         and scope.start_ns() <= launch.start_ns()
                         and launch.end_ns() <= scope.end_ns()]
            enclosing.sort(key=lambda scope: (scope.start_ns(), -scope.end_ns()))
            sources.append(dict(name=launch.name(), thread=launch.device_resource_id(),
                                start_ns=launch.start_ns(), end_ns=launch.end_ns(),
                                enclosing_cpu_scopes=[scope.name() for scope in enclosing]))
        rows.append(dict(name=event.name(), correlation_id=event.correlation_id(),
                         device=event.device_index(), stream=event.device_resource_id(),
                         start_ns=event.start_ns(), end_ns=event.end_ns(),
                         duration_ns=event.duration_ns(), launches=sources))
    return dict(schema='cuda-launch-provenance-v1', diagnostic_only=True, activities=rows)
