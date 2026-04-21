import Foundation

/// Neural TTS. Default: Kokoro (predictor + 3 bucketed decoders by sequence
/// length: 128 / 256 / 512).
///
/// Usage:
/// ```swift
/// let session = try await SpeechSynthesisSession(model: .kokoro)
/// let pcm = try await session.synthesize("Hello world", voice: .afHeart)
/// ```
///
/// - Note: v1-alpha scaffolding. SDK picks the right decoder bucket
///   based on predicted sequence length. Ports from `sample_apps/KokoroDemo`.
public final class SpeechSynthesisSession: CMZSession {

    public enum Model: Sendable {
        case kokoro
        var ids: [String] {
            switch self {
            case .kokoro:
                return ["kokoro_predictor",
                        "kokoro_decoder_128",
                        "kokoro_decoder_256",
                        "kokoro_decoder_512"]
            }
        }
    }

    public enum Voice: String, Sendable, CaseIterable {
        // Kokoro voice IDs.
        case afHeart = "af_heart"
        case afBella = "af_bella"
        case amMichael = "am_michael"
        case bfEmma = "bf_emma"
        case bmGeorge = "bm_george"
    }

    public let model: Model
    public var modelIds: [String] { model.ids }

    public init(model: Model = .kokoro,
                computeUnits: CMZComputeUnits = .auto) async throws {
        self.model = model
        _ = computeUnits
    }

    /// Synthesize a waveform. Returns 24kHz float32 mono PCM.
    public func synthesize(_ text: String, voice: Voice = .afHeart) async throws -> [Float] {
        throw CMZError.inferenceFailed(reason: "SpeechSynthesisSession is scaffolded in v1-alpha")
    }
}
