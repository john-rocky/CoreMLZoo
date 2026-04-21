# CoreMLZoo

A Vision-framework-style Swift SDK for running the CoreML-Models zoo on Apple
platforms. Hides the per-model boilerplate (pre/post-processing, ANE stride
reads, compute-unit selection, seq2seq decoding loops, multi-model pipelines,
tokenizers, DSP) behind high-level `Request` and `Session` types.

```swift
import CoreMLZoo

// Background removal
let out = try await BackgroundRemovalRequest().prepareAndPerform(on: cgImage)
imageView.image = UIImage(cgImage: out.cutout)

// Monocular depth (metric meters)
let depth = try await DepthRequest(model: .moGe2).prepareAndPerform(on: cgImage)

// Florence-2 caption / OCR / free-form Q&A
let session = try await VisionLanguageSession()
let caption = try await session.caption(cgImage, detail: .detailed)
let ocrText = try await session.ocr(cgImage)

// Stable Audio text-to-music
let music = try await TextToMusicSession().generate(
    prompt: "A gentle piano melody with soft strings",
    durationSeconds: 10, steps: 25)
```

Models are **downloaded on first use**, not bundled. The SDK reads the same
HuggingFace-hosted manifest (`models.json`) as the
[CoreML-Models Hub App](https://github.com/john-rocky/CoreML-Models), so any
model you see there is reachable from here.

## Installation

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/john-rocky/CoreMLZoo.git", from: "0.1.0"),
]
```

Minimum platforms: **iOS 17** / **macOS 14** / **visionOS 1** / **macCatalyst 17**.

## API surface

All 18 types fully implemented — pre/post-processing, tokenizers, schedulers,
and state machines are all inside the SDK.

### Stateless `Request` types

| Type | Models | Vision framework equivalent |
|---|---|---|
| `BackgroundRemovalRequest` | RMBG-1.4 | `VNGenerateForegroundInstanceMaskRequest` (lower quality) |
| `DepthRequest` | MoGe-2, Depth Anything V3, MiDaS | — (Apple is LiDAR-only) |
| `UpscaleRequest` | Real-ESRGAN, ESRGAN, UltraSharp, BSRGAN, A-ESRGAN | — |
| `InpaintRequest` | LaMa, AOT-GAN | — |
| `OpenVocabDetectionRequest` | YOLO-World V2-S | `VNRecognizeObjectsRequest` is closed-vocab |
| `FaceEmbeddingRequest` | AdaFace IR-18 | Vision gives landmarks/quality, no identity |
| `Face3DRequest` | 3DDFA V2 | ARKit needs TrueDepth |
| `SourceSeparationRequest` | HTDemucs | — |
| `LowLightEnhanceRequest` | Retinexformer, StableLLVE, Zero-DCE | — |
| `ImageRestorationRequest` | MPRNet, MIRNetv2 variants | — |
| `ColorizeRequest` | DDColor Tiny | — |
| `MusicTranscriptionRequest` | Basic Pitch | — |
| `ZeroShotClassificationRequest` | SigLIP | `VNClassifyImageRequest` is closed-vocab |

### Stateful `Session` types

| Type | Models | Notes |
|---|---|---|
| `VideoMattingSession` | MatAnyone (5 mlpackages) | 768×432 landscape, per-frame ring buffer owned by SDK |
| `VisionLanguageSession` | Florence-2 (3 mlpackages) | caption / detailed / more-detailed / OCR / free-form Q&A |
| `SpeechSynthesisSession` | Kokoro (predictor + 3 bucketed decoders) | 24 kHz mono, voice selection via `.bin` embeddings |
| `VoiceConversionSession` | OpenVoice (speaker encoder + converter) | 22.05 kHz mono, STFT-based speaker embedding |
| `TextToMusicSession` | Stable Audio Open Small (4 mlpackages) | 44.1 kHz stereo, DiT denoise loop + VAE decoder |

## Download UX

Downloads are explicit — the SDK will **not** silently fetch a model mid-inference
(App Review flags surprise network activity). Preload at onboarding:

```swift
for modelId in ["rmbg_1_4", "moge2_vitb_normal_504", "florence2_base"] {
    for try await progress in CMZModelStore.shared.download(id: modelId) {
        updateUI(progress)
    }
}
```

`CMZRequest.prepareAndPerform(on:progress:)` is a convenience that fetches if
missing; prefer explicit `download` for anything over a few MB.

Cache lives under `Application Support/coreml-zoo/models/{id}/`. Files are
SHA-256 verified, zip-unpacked, and marked installed via a `.meta.json`
sentinel. `CMZModelStore.shared.delete(id:)` reclaims the disk.

## Architecture notes

- **Manifest**: fetched from HuggingFace, cached to disk, survives offline launches.
- **Background `URLSession`**: downloads continue while the app is suspended.
- **Stride-aware `MLMultiArray` reads**: ANE pads rows for SIMD alignment, so
  `dataPointer` is not C-contiguous on `.all`/`.cpuAndNeuralEngine`. Shared
  `MLArrayReading` helper uses `vImageConvert_Planar16FtoPlanarF` for FP16
  fast paths.
- **Compute units per model**: auto-resolves from manifest hints. Some are
  load-bearing — MatAnyone's `read` / `read_first` ship `.cpuOnly` because
  the singleton num_objects slice crashes iOS GPU's MPS backend.
- **Tokenizers included**: CLIP byte-level BPE (YOLO-World), BART/GPT-2
  (Florence-2), SentencePiece greedy (SigLIP, T5). No external dependencies.

## License

The SDK itself is MIT (see `LICENSE`). **Model weights are subject to the
licenses of their respective upstream projects** — see the `license` field of
each manifest entry before shipping.

## Related

- [CoreML-Models](https://github.com/john-rocky/CoreML-Models) — the model zoo
  (PyTorch → CoreML conversion scripts, sample apps, manifest)
