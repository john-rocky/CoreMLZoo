# CoreMLZoo

A Vision-framework-style Swift SDK for running the CoreML-Models zoo on Apple
platforms. Hides the per-model boilerplate (pre/post-processing, ANE stride
reads, compute-unit selection, seq2seq decoding loops, multi-model pipelines)
behind high-level `Request` and `Session` types.

```swift
import CoreMLZoo

// Background removal
let req = BackgroundRemovalRequest()
let out = try await req.prepareAndPerform(on: cgImage)
imageView.image = UIImage(cgImage: out.cutout)

// Monocular depth (metric meters)
let depth = try await DepthRequest(model: .moGe2)
    .prepareAndPerform(on: cgImage) { progress in
        print("\(progress.phase) \(Int(progress.fraction * 100))%")
    }
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

## API surface (v1)

Stateless tasks use `Request`; pipelines that keep state between calls use
`Session`.

| Type | Models | Vision framework equivalent |
|---|---|---|
| `BackgroundRemovalRequest` | RMBG-1.4 | `VNGenerateForegroundInstanceMaskRequest` (lower quality) |
| `DepthRequest` | MoGe-2, Depth Anything V3, MiDaS | — (Apple is LiDAR-only) |
| `UpscaleRequest` | Real-ESRGAN, ESRGAN, UltraSharp, SinSR, BSRGAN, A-ESRGAN | — |
| `InpaintRequest` | LaMa, AOT-GAN | — |
| `OpenVocabDetectionRequest` | YOLO-World V2-S | `VNRecognizeObjectsRequest` is closed-vocab |
| `FaceEmbeddingRequest` | AdaFace IR-18 | Vision returns landmarks/quality but no identity |
| `Face3DRequest` | 3DDFA V2 | ARKit needs TrueDepth |
| `SourceSeparationRequest` | HTDemucs | — |
| `LowLightEnhanceRequest` | Retinexformer, StableLLVE, Zero-DCE | — |
| `ImageRestorationRequest` | MPRNet, MIRNetv2 | — |
| `ColorizeRequest` | DDColor Tiny | — |
| `MusicTranscriptionRequest` | Basic Pitch | — |
| `ZeroShotClassificationRequest` | SigLIP | `VNClassifyImageRequest` is closed-vocab |
| `VideoMattingSession` | MatAnyone | `VNGeneratePersonSegmentationRequest` is single-frame |
| `VisionLanguageSession` | Florence-2 (caption / OCR / grounding) | — |
| `SpeechSynthesisSession` | Kokoro TTS | `AVSpeechSynthesizer` (older quality) |
| `VoiceConversionSession` | OpenVoice | — |
| `TextToMusicSession` | Stable Audio Open Small | — |

## Download UX

Downloads are explicit — the SDK will **not** silently fetch a model mid-inference
(App Review flags surprise network activity). Preload at onboarding:

```swift
for modelId in ["rmbg_1_4", "moge2_vitb_normal_504"] {
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

## Status

**v1-alpha**: fully implemented — `BackgroundRemovalRequest`, `DepthRequest`.
Remaining 16 types have the public API surface declared but inference logic
is stubbed (`throw CMZError.inferenceFailed("… scaffolded in v1-alpha")`);
pre/post-processing will be ported from each sample app in
`CoreML-Models/sample_apps/<Name>Demo` as the releases are cut.

Track progress per Request in GitHub issues tagged `v1`.

## License

The SDK itself is MIT (see `LICENSE`). **Model weights are subject to the
licenses of their respective upstream projects** — see the `license` field of
each manifest entry before shipping.

## Related

- [CoreML-Models](https://github.com/john-rocky/CoreML-Models) — the model zoo
  (PyTorch → CoreML conversion scripts, sample apps, manifest)
- [CoreML-Models Hub App](https://apps.apple.com/) — end-user browser for the
  same manifest
