"""Explicit, source-pinned runner exclusions; never a CUDA-coverage certificate.

Only launches outside all model scopes may be excluded. Method annotations
identify reviewed input/metadata/output helpers, not arbitrary phase names.
The one inline boundary is the runner's post-forward hidden-state gather.
"""
from contextlib import contextmanager
from functools import wraps
import hashlib
import os
from pathlib import Path

POLICY = 'runner-boundary-approx-v1'
PREFIX = 'runner_boundary_v1:'
METHODS = ('_update_states', '_prepare_inputs', '_get_slot_mappings',
           '_build_attention_metadata', '_prepare_padding_mask', '_preprocess')


@contextmanager
def runner_boundaries(runner):
    if os.environ.get('PROFILER_RUNNER_POLICY') != POLICY:
        yield
        return
    from vllm.v1.worker import gpu_model_runner as native
    from torch.profiler import record_function
    source = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    if source != os.environ.get('PROFILER_RUNNER_SOURCE_SHA256'):
        raise ValueError('Runner exclusion source differs from reviewed source')
    if type(runner) is not native.GPUModelRunner:
        raise ValueError('Runner exclusions require the reviewed GPUModelRunner')
    restored = []

    def install(target, name, label):
        original = getattr(target, name)
        owned = name in vars(target)
        previous = vars(target).get(name)
        @wraps(original)
        def wrapped(*args, **kwargs):
            with record_function(PREFIX + label):
                return original(*args, **kwargs)
        setattr(target, name, wrapped)
        restored.append((target, name, owned, previous))

    try:
        for name in METHODS:
            install(runner, name, name)
        install(native.AsyncGPUModelRunnerOutput, '__init__', 'async_output')
        yield
    finally:
        for target, name, owned, previous in reversed(restored):
            if owned:
                setattr(target, name, previous)
            else:
                delattr(target, name)


def classify_exclusions(result, provenance, source_sha256):
    """Attach exact unscoped identities; leave every unknown gap unaccepted."""
    by_identity = {}
    for row in provenance['activities']:
        identity = tuple(row[key] for key in
                         ('name', 'correlation_id', 'start_ns', 'end_ns', 'device', 'stream'))
        if identity in by_identity:
            raise ValueError('Duplicate runner provenance identity')
        by_identity[identity] = row
    exclusions = []
    for activity in result['activity_ledger']:
        if activity['location'] != 'unscoped' or activity['claims'] or activity['call'] is not None:
            continue
        row = by_identity[tuple(activity['identity'])]
        if len(row['launches']) != 1:
            raise ValueError('Missing/ambiguous runner launch')
        scopes = row['launches'][0]['enclosing_cpu_scopes']
        if (sum(scope.startswith(('activity_forward_', 'skew_dynamic_forward_')) for scope in scopes) != 1
                or any(scope.startswith('activity_call_') for scope in scopes)):
            raise ValueError('Runner launch overlaps model or lacks forward')
        labels = [scope[len(PREFIX):] for scope in scopes if scope.startswith(PREFIX)]
        label = next((label for label in reversed(labels)
                      if label in (*METHODS, 'async_output')), None)
        if label is None:
            # In vLLM 0.31 this statement is inline in execute_model:
            # sample_hidden_states = hidden_states[logits_indices]. compute_logits
            # is separately model-scoped and is never excluded by this rule.
            phases = [s for s in scopes if s.startswith('gpu_model_runner:')]
            operators = [s for s in scopes if s.startswith('aten::')]
            allowed = (['aten::index'], ['aten::index', 'aten::to', 'aten::_to_copy', 'aten::copy_'])
            kernel = row['name']
            if (phases == ['gpu_model_runner: postprocess'] and operators in allowed
                    and any(s in kernel for s in ('index_elementwise_kernel',
                        'vectorized_gather_kernel', 'direct_copy_kernel_cuda'))):
                label = 'hidden_state_gather'
        if label is not None:
            exclusions.append(dict(identity=activity['identity'], duration_ns=activity['duration_ns'],
                boundary=label, launch=row['launches'][0]))
    evidence = dict(policy=POLICY, runner_source_sha256=source_sha256,
                    exclusions=exclusions, accuracy_validated=False)
    result['audit']['runner_approximation'] = evidence
    return evidence
