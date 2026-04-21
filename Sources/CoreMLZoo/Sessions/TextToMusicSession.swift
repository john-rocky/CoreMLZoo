import Foundation

/// Text → music generation. Default: Stable Audio Open Small
/// (T5 encoder + number embedder + DiT + VAE decoder, ~580 MB FP16).
///
/// - Note: v1-alpha scaffolding. SDK owns the DiT denoising loop and scheduler.
///   Ports from `sample_apps/StableAudioDemo`.
public final class TextToMusicSession: CMZSession {

    public enum Model: Sendable {
        case stableAudioOpenSmall
        /// FP32 variant (1.3 GB) — use on macOS / Pro iPad when FP16 DiT
        /// attention overflows give audible artifacts.
        case stableAudioOpenSmallFP32

        var ids: [String] {
            switch self {
            case .stableAudioOpenSmall:
                return ["stable_audio_t5_encoder",
                        "stable_audio_number_embedder",
                        "stable_audio_dit",
                        "stable_audio_vae_decoder"]
            case .stableAudioOpenSmallFP32:
                return ["stable_audio_t5_encoder",
                        "stable_audio_number_embedder",
                        "stable_audio_dit_fp32",
                        "stable_audio_vae_decoder"]
            }
        }
    }

    public let model: Model
    public var modelIds: [String] { model.ids }

    public init(model: Model = .stableAudioOpenSmall,
                computeUnits: CMZComputeUnits = .auto) async throws {
        self.model = model
        _ = computeUnits
    }

    /// `durationSeconds` must be ≤ 11 (Stable Audio Open Small's training length).
    /// Returns 44.1 kHz stereo interleaved PCM.
    public func generate(prompt: String,
                         durationSeconds: Double = 10,
                         seed: UInt64? = nil,
                         steps: Int = 8) async throws -> [Float] {
        throw CMZError.inferenceFailed(reason: "TextToMusicSession is scaffolded in v1-alpha")
    }
}
