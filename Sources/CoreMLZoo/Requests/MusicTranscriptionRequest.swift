import Foundation
import CoreML
import AVFoundation
import Accelerate

public struct TranscribedNote: Sendable {
    public let startTime: Double  // seconds
    public let endTime: Double
    public let midiPitch: Int     // 21..108 (A0..C8)
    public let confidence: Float
}

/// Audio → MIDI note events. Default: Spotify's Basic Pitch (272 KB,
/// 22050 Hz mono input, fully convolutional with 2s windows at 30-frame
/// overlap).
///
/// Post-processing is a simplified onset-threshold detector that walks
/// each pitch row and emits a note whenever activation crosses
/// `onsetThreshold` for at least 3 consecutive frames. For richer logic
/// (CQT pitch bending, note splitting, polyphonic de-dup) use the raw
/// tensors via `performRaw`.
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

    public struct RawOutput: Sendable {
        /// `(frames, 88)` note activation probabilities in [0, 1].
        public let notes: [[Float]]
        /// `(frames, 88)` onset probabilities in [0, 1].
        public let onsets: [[Float]]
        /// `(frames, 264)` fine-resolution contours (3 bins/semitone).
        public let contours: [[Float]]
        public let framesPerSecond: Double  // 22050 / 256 ≈ 86.13
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
        let raw = try await performRaw(on: input)
        return Self.detectNotes(onsets: raw.onsets,
                                notes: raw.notes,
                                framesPerSecond: raw.framesPerSecond,
                                onsetThreshold: onsetThreshold,
                                frameThreshold: frameThreshold)
    }

    /// Return the raw 3-tensor output (notes / onsets / contours) for
    /// callers that want their own post-processing.
    public func performRaw(on input: Input) async throws -> RawOutput {
        var samples = input.waveform
        if input.sampleRate != BP.sampleRate {
            throw CMZError.invalidInput(reason: "waveform must be 22050 Hz mono (got \(input.sampleRate))")
        }

        // Peak normalize to 0.98 — the upstream model was trained on librosa
        // outputs and iOS CoreAudio MP3 decoders produce slightly hotter
        // samples; normalization keeps the note detector consistent.
        var peak: Float = 0
        vDSP_maxmgv(samples, 1, &peak, vDSP_Length(samples.count))
        if peak > 0 {
            var scale = 0.98 / peak
            vDSP_vsmul(samples, 1, &scale, &samples, 1, vDSP_Length(samples.count))
        }

        let windows = Self.windowAudio(samples)
        let compute = (computeUnits == .auto) ? MLComputeUnits.all
                                              : computeUnits.mlComputeUnits
        let coreModel = try await ModelLoading.load(modelId: modelId, compute: compute)

        var allNotes: [[[Float]]] = [], allOnsets: [[[Float]]] = [], allContours: [[[Float]]] = []
        for window in windows {
            let arr = try MLMultiArray(shape: [1, NSNumber(value: BP.audioNSamples), 1],
                                       dataType: .float32)
            let strides = arr.strides.map { $0.intValue }
            let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
            for j in 0..<window.count { ptr[j * strides[1]] = window[j] }

            let provider = try MLDictionaryFeatureProvider(
                dictionary: ["input_2": MLFeatureValue(multiArray: arr)])
            let output = try await coreModel.prediction(from: provider)

            guard let n = output.featureValue(for: "Identity_1")?.multiArrayValue,
                  let o = output.featureValue(for: "Identity_2")?.multiArrayValue,
                  let c = output.featureValue(for: "Identity")?.multiArrayValue else {
                throw CMZError.inferenceFailed(reason: "missing BasicPitch output")
            }
            allNotes.append(Self.read2D(n, rows: BP.annotFrames, cols: BP.notesBins))
            allOnsets.append(Self.read2D(o, rows: BP.annotFrames, cols: BP.notesBins))
            allContours.append(Self.read2D(c, rows: BP.annotFrames, cols: BP.contoursBins))
        }

        let nOlap = BP.defaultOverlappingFrames / 2  // 15
        return RawOutput(
            notes: Self.unwrap(allNotes, nOlap: nOlap),
            onsets: Self.unwrap(allOnsets, nOlap: nOlap),
            contours: Self.unwrap(allContours, nOlap: nOlap),
            framesPerSecond: Double(BP.sampleRate) / Double(BP.fftHop))
    }

    // MARK: - Constants (matching basic_pitch/constants.py)

    private enum BP {
        static let sampleRate = 22050
        static let fftHop = 256
        static let audioNSamples = 22050 * 2 - 256  // 43844
        static let annotFrames = 172               // 86 * 2
        static let notesBins = 88
        static let contoursBins = 264              // 88 * 3
        static let midiOffset = 21                 // bin 0 == MIDI 21 (A0)
        static let defaultOverlappingFrames = 30
    }

    // MARK: - Windowing + reading

    private static func windowAudio(_ samples: [Float]) -> [[Float]] {
        let overlap = BP.defaultOverlappingFrames * BP.fftHop  // 7680
        let hop = BP.audioNSamples - overlap                   // 36164
        var padded = [Float](repeating: 0, count: overlap / 2)
        padded += samples
        var windows: [[Float]] = []
        var off = 0
        while off < padded.count {
            let end = min(off + BP.audioNSamples, padded.count)
            var w = Array(padded[off..<end])
            if w.count < BP.audioNSamples {
                w += [Float](repeating: 0, count: BP.audioNSamples - w.count)
            }
            windows.append(w)
            off += hop
        }
        return windows
    }

    private static func read2D(_ array: MLMultiArray, rows: Int, cols: Int) -> [[Float]] {
        let strides = array.strides.map { $0.intValue }
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        let s1 = strides.count >= 3 ? strides[1] : cols
        let s2 = strides.count >= 3 ? strides[2] : 1
        var result = [[Float]](); result.reserveCapacity(rows)
        for r in 0..<rows {
            var row = [Float](repeating: 0, count: cols)
            for c in 0..<cols { row[c] = ptr[r * s1 + c * s2] }
            result.append(row)
        }
        return result
    }

    private static func unwrap(_ batches: [[[Float]]], nOlap: Int) -> [[Float]] {
        var out: [[Float]] = []
        for w in batches {
            out.append(contentsOf: w[nOlap..<(w.count - nOlap)])
        }
        return out
    }

    // MARK: - Simple note detection from onsets

    private static func detectNotes(onsets: [[Float]],
                                     notes: [[Float]],
                                     framesPerSecond: Double,
                                     onsetThreshold: Float,
                                     frameThreshold: Float) -> [TranscribedNote] {
        guard !onsets.isEmpty else { return [] }
        let nFrames = onsets.count
        let nPitches = onsets[0].count
        var result: [TranscribedNote] = []

        for p in 0..<nPitches {
            var f = 0
            while f < nFrames {
                if onsets[f][p] > onsetThreshold {
                    let startFrame = f
                    let startConf = onsets[f][p]
                    // Continue while note activation stays above frameThreshold
                    var end = f + 1
                    while end < nFrames && notes[end][p] > frameThreshold {
                        end += 1
                    }
                    if end - startFrame >= 3 {
                        result.append(TranscribedNote(
                            startTime: Double(startFrame) / framesPerSecond,
                            endTime: Double(end) / framesPerSecond,
                            midiPitch: BP.midiOffset + p,
                            confidence: startConf))
                    }
                    f = end + 1
                } else {
                    f += 1
                }
            }
        }
        return result.sorted { $0.startTime < $1.startTime }
    }
}
