import Foundation
import CoreML
import Accelerate

/// Audio source separation. Default: HTDemucs (drums / bass / other / vocals).
///
/// The integrated HTDemucs mlpackage takes a single `audio` input of shape
/// `[1, channels, segmentLength]` and emits 4 float32 stems. The SDK uses
/// the first segment's worth of samples (driven by the model's constraint)
/// and does not chunk — for long tracks, run multiple invocations with
/// overlap-add in user code.
public struct SourceSeparationRequest: CMZRequest {

    public enum Model: String, Sendable, CaseIterable {
        case htDemucs = "htdemucs_source_separation_fp32"

        var sampleRate: Int { 44100 }
        var channels: Int { 2 }
    }

    public struct Stems: Sendable {
        public let drums: [Float]
        public let bass: [Float]
        public let other: [Float]
        public let vocals: [Float]
        public let sampleRate: Int
        public let channels: Int
    }

    public struct Input: Sendable {
        /// Stereo-interleaved (or mono) waveform.
        public var waveform: [Float]
        public var sampleRate: Int
        public var isStereo: Bool
        public init(waveform: [Float], sampleRate: Int = 44100, isStereo: Bool = true) {
            self.waveform = waveform
            self.sampleRate = sampleRate
            self.isStereo = isStereo
        }
    }

    public let model: Model
    public var computeUnits: CMZComputeUnits

    public init(model: Model = .htDemucs, computeUnits: CMZComputeUnits = .auto) {
        self.model = model; self.computeUnits = computeUnits
    }

    public var modelId: String { model.rawValue }

    public func perform(on input: Input) async throws -> Stems {
        guard input.sampleRate == model.sampleRate else {
            throw CMZError.invalidInput(
                reason: "HTDemucs expects \(model.sampleRate) Hz (got \(input.sampleRate))")
        }

        let compute = (computeUnits == .auto) ? MLComputeUnits.all
                                              : computeUnits.mlComputeUnits
        let coreModel = try await ModelLoading.load(modelId: modelId, compute: compute)

        let desc = coreModel.modelDescription.inputDescriptionsByName
        guard let firstInput = desc.values.first(where: { $0.type == .multiArray }),
              let constraint = firstInput.multiArrayConstraint else {
            throw CMZError.inferenceFailed(reason: "expected multiarray input")
        }
        let shape = constraint.shape.map { $0.intValue }
        let segmentLength = shape.last ?? input.waveform.count
        let modelChannels = shape.count >= 2 ? shape[shape.count - 2] : model.channels

        let inputArr = try MLMultiArray(shape: constraint.shape, dataType: .float32)
        memset(inputArr.dataPointer, 0, segmentLength * modelChannels * 4)
        let ptr = inputArr.dataPointer.assumingMemoryBound(to: Float.self)

        // De-interleave stereo (or duplicate mono) into planar channels.
        let n = min(segmentLength, input.isStereo ? input.waveform.count / 2
                                                  : input.waveform.count)
        if input.isStereo {
            for i in 0..<n {
                ptr[0 * segmentLength + i] = input.waveform[i * 2]
                if modelChannels > 1 {
                    ptr[1 * segmentLength + i] = input.waveform[i * 2 + 1]
                }
            }
        } else {
            for i in 0..<n {
                ptr[0 * segmentLength + i] = input.waveform[i]
                if modelChannels > 1 {
                    ptr[1 * segmentLength + i] = input.waveform[i]
                }
            }
        }

        let inputName = desc.keys.first ?? "audio"
        let output: MLFeatureProvider
        do {
            output = try await coreModel.prediction(from:
                MLDictionaryFeatureProvider(dictionary: [inputName: inputArr]))
        } catch {
            throw CMZError.inferenceFailed(reason: "HTDemucs: \(error)")
        }

        // HTDemucs output feature names are the stem names; order may vary.
        let stemOrder = ["drums", "bass", "other", "vocals"]
        var stemData: [String: [Float]] = [:]
        for name in output.featureNames {
            if let arr = output.featureValue(for: name)?.multiArrayValue {
                let floats = Self.readFloats(arr)
                // Try to match by feature name; fallback to positional order later.
                let key = stemOrder.first(where: { name.lowercased().contains($0) }) ?? name
                stemData[key] = floats
            }
        }

        // If feature names didn't match the canonical order, assume positional
        // order from the output dictionary.
        if stemData.count == output.featureNames.count,
           !stemOrder.allSatisfy({ stemData[$0] != nil }) {
            stemData = [:]
            for (i, name) in output.featureNames.enumerated() {
                if let arr = output.featureValue(for: name)?.multiArrayValue, i < stemOrder.count {
                    stemData[stemOrder[i]] = Self.readFloats(arr)
                }
            }
        }
        return Stems(
            drums: stemData["drums"] ?? [],
            bass: stemData["bass"] ?? [],
            other: stemData["other"] ?? [],
            vocals: stemData["vocals"] ?? [],
            sampleRate: model.sampleRate,
            channels: modelChannels)
    }

    private static func readFloats(_ array: MLMultiArray) -> [Float] {
        let n = array.count
        var out = [Float](repeating: 0, count: n)
        if array.dataType == .float16 {
            let src = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            out.withUnsafeMutableBufferPointer { dst in
                var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src),
                                      height: 1, width: vImagePixelCount(n), rowBytes: n * 2)
                var d = vImage_Buffer(data: dst.baseAddress!,
                                      height: 1, width: vImagePixelCount(n), rowBytes: n * 4)
                vImageConvert_Planar16FtoPlanarF(&s, &d, 0)
            }
        } else {
            memcpy(&out, array.dataPointer, n * 4)
        }
        return out
    }
}
