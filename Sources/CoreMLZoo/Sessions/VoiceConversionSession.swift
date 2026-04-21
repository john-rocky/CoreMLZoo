import Foundation
import CoreML
import Accelerate

/// OpenVoice voice conversion. Session holds both mlpackages (speaker
/// encoder + voice converter) and exposes the canonical two-step workflow:
/// extract a 256-d speaker embedding from a reference waveform, then
/// convert any source waveform to that target speaker.
///
/// All audio is 22050 Hz mono. The SDK does STFT (nFFT=1024, hop=256,
/// window=1024) internally.
public final class VoiceConversionSession: CMZSession {

    public enum Model: Sendable {
        case openVoice
        var id: String { "openvoice" }
    }

    public static let sampleRate = 22050
    static let nFFT = 1024
    static let hopLen = 256
    static let winLen = 1024
    static let freqBins = nFFT / 2 + 1  // 513
    static let embDim = 256

    public let model: Model
    public var modelIds: [String] { [model.id] }

    private let speakerEncoder: MLModel
    private let voiceConverter: MLModel

    public init(model: Model = .openVoice,
                computeUnits: CMZComputeUnits = .auto) async throws {
        self.model = model
        let modelId = model.id
        let compute = (computeUnits == .auto) ? MLComputeUnits.all
                                              : computeUnits.mlComputeUnits

        async let se = Self.load(modelId: modelId, substring: "speaker", compute: compute)
        async let vc = Self.load(modelId: modelId, substring: "converter", compute: compute)
        self.speakerEncoder = try await se
        self.voiceConverter = try await vc
    }

    // MARK: - Public API

    /// Extract a 256-d speaker embedding from a reference waveform.
    public func extractSpeakerEmbedding(from waveform: [Float],
                                         sampleRate: Int = sampleRate) async throws -> [Float] {
        guard sampleRate == Self.sampleRate else {
            throw CMZError.invalidInput(
                reason: "OpenVoice requires \(Self.sampleRate) Hz (got \(sampleRate))")
        }
        let spec = waveform.withUnsafeBufferPointer { buf -> [Float] in
            guard let base = buf.baseAddress else { return [] }
            return STFT.magnitude(samples: base, count: waveform.count,
                                   nFFT: Self.nFFT, hopLength: Self.hopLen,
                                   winLength: Self.winLen)
        }
        let numFrames = spec.count / Self.freqBins

        let input = try MLMultiArray(shape: [1, NSNumber(value: numFrames),
                                              NSNumber(value: Self.freqBins)],
                                      dataType: .float32)
        let ptr = input.dataPointer.assumingMemoryBound(to: Float.self)
        for t in 0..<numFrames {
            for f in 0..<Self.freqBins {
                ptr[t * Self.freqBins + f] = spec[f * numFrames + t]  // transpose
            }
        }
        let output = try await speakerEncoder.prediction(from:
            MLDictionaryFeatureProvider(dictionary: [
                "spectrogram": MLFeatureValue(multiArray: input)
            ]))
        guard let emb = output.featureValue(for: "speaker_embedding")?.multiArrayValue else {
            throw CMZError.inferenceFailed(reason: "speaker_embedding missing")
        }
        return Self.readFloats(emb)
    }

    /// Convert `sourceWaveform` to sound like the target speaker.
    public func convert(_ sourceWaveform: [Float],
                        to targetEmbedding: [Float],
                        sourceEmbedding: [Float]? = nil,
                        sampleRate: Int = sampleRate) async throws -> [Float] {
        guard sampleRate == Self.sampleRate else {
            throw CMZError.invalidInput(
                reason: "OpenVoice requires \(Self.sampleRate) Hz (got \(sampleRate))")
        }

        // If caller didn't supply a source embedding, extract one so we're
        // doing real tone-color conversion (not self → self).
        let srcEmb: [Float]
        if let provided = sourceEmbedding {
            srcEmb = provided
        } else {
            srcEmb = try await extractSpeakerEmbedding(from: sourceWaveform,
                                                       sampleRate: sampleRate)
        }

        let spec = sourceWaveform.withUnsafeBufferPointer { buf -> [Float] in
            guard let base = buf.baseAddress else { return [] }
            return STFT.magnitude(samples: base, count: sourceWaveform.count,
                                   nFFT: Self.nFFT, hopLength: Self.hopLen,
                                   winLength: Self.winLen)
        }
        let numFrames = spec.count / Self.freqBins

        let specArr = try MLMultiArray(shape: [1, NSNumber(value: Self.freqBins),
                                                NSNumber(value: numFrames)],
                                        dataType: .float32)
        let sp = specArr.dataPointer.assumingMemoryBound(to: Float.self)
        memcpy(sp, spec, Self.freqBins * numFrames * 4)

        let specLen = try MLMultiArray(shape: [1], dataType: .float32)
        specLen[0] = NSNumber(value: numFrames)

        let srcSpk = try MLMultiArray(shape: [1, NSNumber(value: Self.embDim), 1],
                                       dataType: .float32)
        let tgtSpk = try MLMultiArray(shape: [1, NSNumber(value: Self.embDim), 1],
                                       dataType: .float32)
        let srcP = srcSpk.dataPointer.assumingMemoryBound(to: Float.self)
        let tgtP = tgtSpk.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<min(Self.embDim, srcEmb.count) { srcP[i] = srcEmb[i] }
        for i in 0..<min(Self.embDim, targetEmbedding.count) { tgtP[i] = targetEmbedding[i] }

        let output = try await voiceConverter.prediction(from:
            MLDictionaryFeatureProvider(dictionary: [
                "spectrogram": MLFeatureValue(multiArray: specArr),
                "spec_lengths": MLFeatureValue(multiArray: specLen),
                "source_speaker": MLFeatureValue(multiArray: srcSpk),
                "target_speaker": MLFeatureValue(multiArray: tgtSpk),
            ]))

        // OpenVoice output feature name varies — take the first multi-array.
        guard let audioArr = output.featureNames
            .compactMap({ output.featureValue(for: $0)?.multiArrayValue })
            .first else {
            throw CMZError.inferenceFailed(reason: "converter output missing")
        }
        var audio = Self.readFloats(audioArr)
        let peak = audio.map { abs($0) }.max() ?? 1
        if peak > 1 {
            var s: Float = 0.95 / peak
            vDSP_vsmul(audio, 1, &s, &audio, 1, vDSP_Length(audio.count))
        }
        return audio
    }

    // MARK: - Loading

    private static func load(modelId: String, substring: String,
                              compute: MLComputeUnits) async throws -> MLModel {
        let dir = CMZPaths.modelDir(id: modelId)
        guard FileManager.default.fileExists(atPath: CMZPaths.metaFile(modelId: modelId).path) else {
            throw CMZError.modelNotInstalled(id: modelId)
        }
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)) ?? []
        guard let url = entries.first(where: {
            ($0.pathExtension == "mlpackage" || $0.pathExtension == "mlmodelc")
                && $0.lastPathComponent.lowercased().contains(substring)
        }) else {
            throw CMZError.inferenceFailed(reason: "OpenVoice sub-model '\(substring)' missing")
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = compute
        return try await ModelLoading.loadCompiled(at: url, configuration: cfg)
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
