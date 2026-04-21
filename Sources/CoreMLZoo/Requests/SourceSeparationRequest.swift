import Foundation

/// Audio source separation. Default: HTDemucs (drums / bass / other / vocals).
///
/// - Note: v1-alpha scaffolding. Ports from `sample_apps/DemucsDemo`.
public struct SourceSeparationRequest: CMZRequest {

    public enum Model: String, Sendable, CaseIterable {
        case htDemucs = "htdemucs_source_separation_fp32"
    }

    public struct Stems: Sendable {
        public let drums: [Float]
        public let bass: [Float]
        public let other: [Float]
        public let vocals: [Float]
        /// Sample rate in Hz (HTDemucs = 44100).
        public let sampleRate: Int
    }

    public struct Input: Sendable {
        public var waveform: [Float]   // mono or stereo-interleaved
        public var sampleRate: Int
        public var isStereo: Bool
        public init(waveform: [Float], sampleRate: Int = 44100, isStereo: Bool = true) {
            self.waveform = waveform; self.sampleRate = sampleRate; self.isStereo = isStereo
        }
    }

    public let model: Model
    public var computeUnits: CMZComputeUnits

    public init(model: Model = .htDemucs, computeUnits: CMZComputeUnits = .auto) {
        self.model = model; self.computeUnits = computeUnits
    }

    public var modelId: String { model.rawValue }

    public func perform(on input: Input) async throws -> Stems {
        throw CMZError.inferenceFailed(reason: "SourceSeparationRequest is scaffolded in v1-alpha")
    }
}
