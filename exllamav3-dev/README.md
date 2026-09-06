# Local ExLlamaV3 build workspace

This directory is a machine-local build workspace for the optional Tutor LLM
provider. It is not part of IPA's core runtime and model weights are stored under
`models/`, outside the public repository surface.

## Layout

```text
exllamav3-dev/
├── source/       ExLlamaV3 checkout, setup and source dependencies
├── build/        locally compiled `exllamav3_ext*.pyd`
├── tools/        local speculative-decoding/diagnostic utilities
└── README.md
```

The IPA provider auto-discovers compiled extensions in `build/`, then the legacy
workspace locations for compatibility. The provider remains importable without
loading the GPU model; Tutor execution requires the optional Tutor profile,
CUDA-compatible PyTorch and a compiled extension for the local GPU.

## Compile

From the project root, follow the commands in `AGENTS.md`. Compile from
`exllamav3-dev/source/` with the local Python/CUDA toolchain, then place the
resulting `exllamav3_ext*.pyd` in `exllamav3-dev/build/`.

Do not commit model weights, compiled binaries, caches or local benchmark output.
