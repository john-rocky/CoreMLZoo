import Foundation

/// Voice conversion. Default: OpenVoice (speaker encoder + converter).
///
/// - Note: v1-alpha scaffolding. Ports from `sample_apps/OpenVoiceDemo`.
public final class VoiceConversionSession: CMZSession {

    public enum Model: Sendable {
        case openVoice
        var ids: [String] {
            switch self {
            case .openVoice:
                return ["openvoice_speaker_encoder", "openvoice_voice_converter"]
            }
        }
    }

    public let model: Model
    public var modelIds: [String] { model.ids }

    public init(model: Model = .openVoice,
                computeUnits: CMZComputeUnits = .auto) async throws {
        self.model = model
        _ = computeUnits
    }

    /// Extract a 256-dim speaker embedding from a reference waveform.
    public func extractSpeakerEmbedding(_ referenceWaveform: [Float],
                                         sampleRate: Int = 22050) async throws -> [Float] {
        throw CMZError.inferenceFailed(reason: "VoiceConversionSession is scaffolded in v1-alpha")
    }

    /// Convert `sourceWaveform` to sound like the target speaker embedding.
    public func convert(_ sourceWaveform: [Float],
                        to targetEmbedding: [Float],
                        sampleRate: Int = 22050) async throws -> [Float] {
        throw CMZError.inferenceFailed(reason: "VoiceConversionSession is scaffolded in v1-alpha")
    }
}
