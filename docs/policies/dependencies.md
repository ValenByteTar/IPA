# Dependency policy

IPA keeps the core dependency surface small and installs optional capabilities by
profile. Optional dependencies must be lazy-imported, version constrained and
reported in experiment metadata.

Rules:

- no floating `latest` dependencies;
- benchmark a tool before making it preferred;
- preserve adapter boundaries;
- record license, deployment mode and version;
- do not add a service when a local implementation is adequate;
- keep GPU, browser, OCR, vector and Tutor dependencies out of core imports;
- keep JSON Schema validation dev/test-only unless runtime validation explicitly
  requires it.
