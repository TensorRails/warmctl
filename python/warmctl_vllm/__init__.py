"""warmctl's vLLM worker extension.

The host loads `warmctl_vllm.worker.Extension` with `--worker-extension-cls`
and drives it through `collective_rpc`. Everything here is about one thing:
the sleep-mode allocator's weights pool, exported as bytes before the
process snapshot and reloaded to the same addresses after the restore.
"""
