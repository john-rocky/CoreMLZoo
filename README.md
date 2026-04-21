# CoreMLZoo

[![CI](https://github.com/john-rocky/CoreMLZoo/actions/workflows/ci.yml/badge.svg)](https://github.com/john-rocky/CoreMLZoo/actions/workflows/ci.yml)

A Vision-framework-style Swift SDK for running the
[CoreML-Models](https://github.com/john-rocky/CoreML-Models) zoo on Apple
platforms. Hides the per-model boilerplate (pre/post-processing, ANE stride
reads, compute-unit selection, seq2seq decoding loops, multi-model pipelines,
tokenizers, DSP) behind high-level `Request` and `Session` types.

```swift
import CoreMLZoo

// Background removal
let out = try await BackgroundRemovalRequest().prepareAndPerform(on: cgImage)
imageView.image = UIImage(cgImage: out.cutout)

// Monocular depth (metric meters via MoGe-2)
let depth = try await DepthRequest(model: .moGe2).prepareAndPerform(on: cgImage)

// Florence-2: caption / OCR / free-form Q&A
let vision = try await VisionLanguageSession()
let caption = try await vision.caption(cgImage, detail: .detailed)
let text    = try await vision.ocr(cgImage)

// Stable Audio text-to-music
let music = try await TextToMusicSession().generate(
    prompt: "A gentle piano melody with soft strings",
    durationSeconds: 10, steps: 25)
```

Models are **downloaded on first use**, not bundled. The SDK reads the
HuggingFace-hosted manifest at
`huggingface.co/mlboydaisuke/coreml-zoo/resolve/main/models.json`, the same
one the official Hub App uses — any model you see there is reachable from
here.

## Installation

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/john-rocky/CoreMLZoo.git", from: "0.1.0"),
]
```

Minimum platforms: **iOS 17** / **macOS 14** / **visionOS 1** / **macCatalyst 17**.

## API surface

14 task types, all backed by models in the default HF manifest and all built
on every commit (see `.github/workflows/ci.yml`).

### Stateless `Request` types

| Type | Manifest model | Vision framework equivalent |
|---|---|---|
| `BackgroundRemovalRequest` | `rmbg_1_4` | `VNGenerateForegroundInstanceMaskRequest` (lower quality) |
| `DepthRequest` | `moge2_vitb_normal_504`, `depth_anything_v3_small_504`, `depth_anything_v3_base_504` | — (Apple is LiDAR-only) |
| `UpscaleRequest` | `realesrgan`, `gfpgan` | — |
| `InpaintRequest` | `lama` | — |
| `OpenVocabDetectionRequest` | `yoloworld` | `VNRecognizeObjectsRequest` is closed-vocab |
| `Face3DRequest` | `face3d` (3DDFA V2) | ARKit requires TrueDepth |
| `SourceSeparationRequest` | `demucs` (HTDemucs) | — |
| `ColorizeRequest` | `ddcolor` | — |
| `ZeroShotClassificationRequest` | `siglip` | `VNClassifyImageRequest` is closed-vocab |

### Stateful `Session` types

| Type | Manifest model | Notes |
|---|---|---|
| `VideoMattingSession` | `matanyone` (5 mlpackages) | 768×432 landscape, per-frame ring buffer owned by SDK |
| `VisionLanguageSession` | `florence2` (3 mlpackages + vocab) | caption / detailed / OCR / free-form Q&A |
| `SpeechSynthesisSession` | `kokoro` (4 mlpackages + vocab + voice blobs) | 24 kHz mono, bucketed decoder (128/256/512). **English TTS quality requires pre-computed phonemes — see caveat below** |
| `VoiceConversionSession` | `openvoice` (2 mlpackages) | 22.05 kHz mono, built-in STFT |
| `TextToMusicSession` | `stable_audio` (4 mlpackages + vocab) | 44.1 kHz stereo, DiT denoise loop + VAE |

## Download UX

Downloads are explicit — the SDK will **not** silently fetch a model
mid-inference (App Review flags surprise network activity). Preload at
onboarding:

```swift
for id in ["rmbg_1_4", "moge2_vitb_normal_504", "florence2"] {
    for try await progress in CMZModelStore.shared.download(id: id) {
        updateUI(progress)
    }
}
```

`CMZRequest.prepareAndPerform(on:progress:)` is a convenience that fetches
if missing; prefer explicit `download` for anything over a few MB.

Cache lives under `Application Support/coreml-zoo/models/{id}/`. Files are
SHA-256 verified, zip-unpacked, and marked installed via a `.meta.json`
sentinel. `CMZModelStore.shared.delete(id:)` reclaims the disk.

## Known caveats

1. **`SpeechSynthesisSession.synthesize(_:)` uses a minimal G2P (just
   lowercases the input)** which produces poor quality on English. For
   production TTS, phonemize externally (e.g. espeak, piper-phonemize,
   whisper-phonemize) and call `synthesizePhonemes(_:voice:)` directly.
   Japanese and other languages with phonemic orthography work better out
   of the box.
2. **`VideoMattingSession` is locked to 768×432 landscape.** Portrait
   sources must be rotated by the caller before `process(_:)`.
3. **`UpscaleRequest` reads the compiled mlpackage's fixed input size**
   — callers must pre-resize/crop to the tile size the converter used
   (typically 256×256 or 512×512).

## Architecture notes

- **Manifest**: fetched from HuggingFace, cached to disk, survives offline launches.
- **Background `URLSession`**: downloads continue while the app is suspended.
- **Stride-aware `MLMultiArray` reads**: ANE pads rows for SIMD alignment,
  so `dataPointer` is not C-contiguous on `.all`/`.cpuAndNeuralEngine`.
  Shared `MLArrayReading` helper uses `vImageConvert_Planar16FtoPlanarF`
  for FP16 fast paths.
- **Compute units per model**: auto-resolves from manifest hints. Some are
  load-bearing — MatAnyone's `read` / `read_first` ship `.cpuOnly` because
  the singleton num_objects slice crashes iOS GPU's MPS backend.
- **Tokenizers included**: CLIP byte-level BPE (YOLO-World), BART/GPT-2
  (Florence-2), SentencePiece greedy (SigLIP, T5). No external dependencies.

## License

The SDK itself is MIT (see `LICENSE`). **Model weights are subject to the
licenses of their respective upstream projects** — see the `license` field
of each manifest entry before shipping.

## Related

- [CoreML-Models](https://github.com/john-rocky/CoreML-Models) — the model
  zoo (PyTorch → CoreML conversion scripts, sample apps, manifest)
- [CoreML-Models Hub App](https://github.com/john-rocky/CoreML-Models/tree/master/sample_apps/CoreMLModelsApp)
  — end-user browser for the same manifest
