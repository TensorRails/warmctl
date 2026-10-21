# vLLM recipe image with libcuda 615

`registry.warmctl.io/warmctl/vllm-openai:0.31-dsv41-b300-cu13compat615`
(`sha256:64ed59eefab298ecf4a403d5d6ff73ebac7cb70a5c0efde0cc8c5937fcf831cb`)
is vLLM's DeepSeek-V4.1-Flash B300 recipe image
(`vllm/vllm-openai:nightly-x86_64-dev-20261012-59e75f4`, vLLM 0.31) with one
change: its CUDA forward-compatibility directory holds libcuda 615.71.09,
taken from vLLM's CUDA 13.4 image, instead of 580.95.05. The NVIDIA container
runtime then loads 615 on a host with an r580 kernel driver.

An r580 libcuda cannot add devices to multicast objects in a process restored
by cuda-checkpoint (ai-dynamo/snapshot#510); r595 and newer can. The
`deepseek-v41-flash-1017-b300-dep8` profile uses DeepGEMM mega-MoE, whose
buffers are multicast objects, so its snapshot only restores on this image.

Build and publish it as `RELEASING.md` publishes the runtime image:

```bash
docker build -t localhost:5000/warmctl/vllm-openai:0.31-dsv41-b300-cu13compat615 deploy/images/vllm-cu13compat615
docker push localhost:5000/warmctl/vllm-openai:0.31-dsv41-b300-cu13compat615
regctl image copy localhost:5000/warmctl/vllm-openai:0.31-dsv41-b300-cu13compat615 registry.warmctl.io/warmctl/vllm-openai:0.31-dsv41-b300-cu13compat615
```
