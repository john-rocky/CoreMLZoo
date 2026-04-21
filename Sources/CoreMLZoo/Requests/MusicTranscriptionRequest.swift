import Foundation

public struct TranscribedNote: Sendable {
    public let startTime: Double
    public let endTime: Double
    public let midiPitch: Int
    public let confidence: Float
}

/// Audio → MIDI note events. Default: Basic Pitch (Spotify, 272 KB).
///
/// - Note: v1-alpha scaffolding. Ports from `sample_apps/BasicPitchDemo`.
///   Post-processing is the non-trivial part: the "nmp" output needs
///   stride-aware reading, threshold-based onset/offset detection.
public struct MusicTranscriptionRequest: CMZRequest {

    public enum Model: String, Sendable, CaseIterable {
        case basicPitch = "basic_pitch_nmp"
    }

    public struct Input: Sendable {
        public var waveform: [Float]
        public var sampleRate: Int
        public init(waveform: [Float], sampleRate: Int = 22050) {
            self.waveform = waveform; self.sampleRate = sampleRate
        }
    }

    public let model: Model
    public var computeUnits: CMZComputeUnits
    public var onsetThreshold: Float = 0.5
    public var frameThreshold: Float = 0.3

    public init(model: Model = .basicPitch, computeUnits: CMZComputeUnits = .auto) {
        self.model = model; self.computeUnits = computeUnits
    }

    public var modelId: String { model.rawValue }

    public func perform(on input: Input) async throws -> [TranscribedNote] {
        throw CMZError.inferenceFailed(reason: "MusicTranscriptionRequest is scaffolded in v1-alpha")
    }
}
