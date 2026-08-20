# ZS32 Demo Runtime Speedup Design

## Goal

Reduce the elapsed time and sustained CPU/disk load of the eight-view Demo
without changing image pixels, Template/PatchCore/YOLO decisions, thresholds,
evidence semantics, or the global Template gate.

## Design

1. Dashboard source images are decoded once per unique immutable result path and
   reused by parser validation and every later render. Cached arrays are never
   returned directly to callers that may mutate them.
2. Stage35 validates manifest identity, topology and file layout without decoding
   all eight images. The resident worker remains the single authoritative image
   decoder and validates exact dimensions.
3. The Demo runtime copies each already-encoded source PNG byte-for-byte into the
   immutable result, creates ROI crops in memory, and passes arrays directly to
   Template. Crop PNGs are materialized only after every Template passes, because
   only PatchCore requires paths. A Template-NG result therefore contains no
   unnecessary crop PNG archive.
4. The successful HDR path keeps both operator confirmations and all image-quality
   settings, but changes the grouped trigger interval from 0.5 s to the repository
   validated 0.2 s. Retry and exposure behavior remain unchanged.
5. PatchCore concurrency is benchmark-gated. Distinct models/engines may use a
   bounded two-worker scheduler only if repeated same-image GPU comparison proves
   identical scores, raw maps, masks and evidence pixels and lowers wall time.
   Otherwise the serial implementation remains authoritative.

## Failure behavior

- Bad or dimensionally invalid PNGs fail in the worker before result publication.
- Source-copy or crop-persistence failures remain fail-closed runtime errors.
- Cache entries are bounded and keyed by immutable resolved result paths.
- No camera confirmation is removed or automated.

## Verification

- Each behavior is introduced with a failing regression test.
- Run focused Dashboard, live commissioning, Demo runtime and model-runtime tests.
- Compare the same saved eight-view sample before and after, including manifest
  decisions and source PNG SHA-256.
- Run a real GPU PatchCore A/B before enabling any concurrent scheduler.

