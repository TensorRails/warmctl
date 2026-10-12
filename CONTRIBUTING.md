# Contributing

warmctl stays small on purpose: one mechanism per problem, no per-model or
per-GPU code where a profile or a hardware row can say it, and only the
checks that catch a failure a restore would otherwise hide. A change that
adds code explains which failure it prevents and how a test shows it.

## Before a pull request

```
make test pytest profiles-check lint
```

`make criu-test e2e-test` need Docker and root; run them for changes to the
agent, the CRIU or CUDA steps, the store or the registry code.

## Profiles

A new model is a YAML file in `profiles/` (see the README). It is added once
it has baked and restored on the hardware it names, with the restore time in
the pull request.

## GPU changes

Anything the CPU tests stand in for (CUDA, the weights pool, cuInterpose,
vLLM's sleep mode) is proven on a GPU before it is merged:
[docs/gpu-runbook.md](docs/gpu-runbook.md) describes the acceptance lanes, and
the pull request carries their restore times.

## Style

Go: gofmt, golangci-lint (`.golangci.yml`). Python: ruff, no runtime
dependencies outside vLLM's image. Comments say why, not what.
