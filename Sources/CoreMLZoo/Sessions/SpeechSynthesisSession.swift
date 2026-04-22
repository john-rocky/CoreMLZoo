import Foundation
import CoreML

/// Neural TTS. Default: Kokoro (predictor + 3 bucketed decoders by sequence
/// length 128 / 256 / 512). The SDK picks the smallest bucket that fits the
/// predicted duration.
///
/// Voices ship as `voice_<id>.bin` float32 blobs in the model directory.
/// The phoneme vocab ships as `kokoro_vocab.json`. Both are bundled as
/// additional files in the manifest entry.
///
/// The built-in G2P is deliberately minimal — it passes lowercased text
/// through to the vocab lookup. For high-quality English or Japanese
/// phonemization, tokenize externally (e.g. via an espeak bridge) and pass
/// the phoneme string via `synthesizePhonemes(_:)`.
public final class SpeechSynthesisSession: CMZSession {

    public enum Model: Sendable {
        case kokoro
        var id: String { "kokoro" }
    }

    public struct Voice: Sendable {
        public let id: String
        public init(_ id: String) { self.id = id }
        // Common Kokoro voices.
        public static let afHeart   = Voice("af_heart")
        public static let afBella   = Voice("af_bella")
        public static let amMichael = Voice("am_michael")
        public static let bfEmma    = Voice("bf_emma")
        public static let bmGeorge  = Voice("bm_george")
    }

    public let model: Model
    public var modelIds: [String] { [model.id] }

    public static let sampleRate = 24000

    private let predictor: MLModel
    private let decoders: [Int: MLModel]  // bucket size → decoder
    private let vocab: [String: Int32]

    public init(model: Model = .kokoro,
                computeUnits: CMZComputeUnits = .auto) async throws {
        self.model = model
        let modelId = model.id
        let compute = (computeUnits == .auto) ? MLComputeUnits.all
                                              : computeUnits.mlComputeUnits

        self.predictor = try await Self.loadNamed(modelId: modelId,
                                                   substring: "predict",
                                                   compute: compute)
        var decoders: [Int: MLModel] = [:]
        for bucket in [128, 256, 512] {
            decoders[bucket] = try await Self.loadNamed(modelId: modelId,
                                                        substring: "decoder_\(bucket)",
                                                        compute: compute)
        }
        self.decoders = decoders
        self.vocab = try Self.loadVocab(modelId: modelId)
    }

    // MARK: - Public API

    /// Synthesize `text` as 24 kHz float mono PCM using `voice`.
    ///
    /// > Warning: the built-in text → phoneme step is minimal (it just
    /// > lowercases the input). English output will be poor unless you
    /// > pre-phonemize with an external G2P (espeak / piper-phonemize)
    /// > and use `synthesizePhonemes(_:voice:)` instead.
    public func synthesize(_ text: String, voice: Voice = .afHeart) async throws -> [Float] {
        try await synthesizePhonemes(text.lowercased(), voice: voice)
    }

    /// Skip the built-in (minimal) G2P and pass phoneme characters directly.
    public func synthesizePhonemes(_ phonemes: String,
                                    voice: Voice = .afHeart) async throws -> [Float] {
        var tokenIds: [Int32] = [0]
        for ch in phonemes.unicodeScalars {
            if let id = vocab[String(ch)] { tokenIds.append(id) }
        }
        tokenIds.append(0)
        if tokenIds.count > 256 { tokenIds = Array(tokenIds.prefix(255)) + [0] }
        let T = tokenIds.count

        // Load voice embedding (512×256 float32 blob; row per token count).
        let refS = try Self.loadVoice(modelId: model.id, id: voice.id, tokenCount: T)
        let refSStyle = Array(refS[128..<256])

        // Predictor
        let inputIds = try MLMultiArray(shape: [1, NSNumber(value: T)], dataType: .int32)
        let idPtr = inputIds.dataPointer.assumingMemoryBound(to: Int32.self)
        for (i, t) in tokenIds.enumerated() { idPtr[i] = t }

        let refStyleArr = try MLMultiArray(shape: [1, 128], dataType: .float32)
        let rsPtr = refStyleArr.dataPointer.assumingMemoryBound(to: Float.self)
        for (i, v) in refSStyle.enumerated() { rsPtr[i] = v }

        let predOutput = try await predictor.prediction(from:
            MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: inputIds),
                "ref_s_style": MLFeatureValue(multiArray: refStyleArr)
            ]))
        guard let durArr = predOutput.featureValue(for: "duration")?.multiArrayValue,
              let dForAlign = predOutput.featureValue(for: "d_for_align")?.multiArrayValue,
              let tEn = predOutput.featureValue(for: "t_en")?.multiArrayValue else {
            throw CMZError.inferenceFailed(reason: "Kokoro predictor output missing")
        }

        // Expand duration to integer frames
        var predDur = [Int](repeating: 0, count: T)
        var totalFrames = 0
        for i in 0..<T {
            predDur[i] = max(1, Int(durArr[i].floatValue.rounded()))
            totalFrames += predDur[i]
        }

        // Bucket routing
        let buckets = [128, 256, 512]
        let bucket = buckets.first { $0 >= totalFrames } ?? buckets.last!
        guard let decoder = decoders[bucket] else {
            throw CMZError.inferenceFailed(reason: "decoder bucket \(bucket) missing")
        }

        // Repeat-interleave expansion (en + asr).
        let dHidden = 640, tHidden = 512
        let enAligned = try MLMultiArray(shape: [1, NSNumber(value: dHidden),
                                                    NSNumber(value: bucket)],
                                         dataType: .float32)
        let asrAligned = try MLMultiArray(shape: [1, NSNumber(value: tHidden),
                                                     NSNumber(value: bucket)],
                                          dataType: .float32)
        let enPtr = enAligned.dataPointer.assumingMemoryBound(to: Float.self)
        let asrPtr = asrAligned.dataPointer.assumingMemoryBound(to: Float.self)
        memset(enPtr, 0, dHidden * bucket * 4)
        memset(asrPtr, 0, tHidden * bucket * 4)

        let dStrides = dForAlign.strides.map { $0.intValue }
        let tStrides = tEn.strides.map { $0.intValue }
        let dPtr = dForAlign.dataPointer.assumingMemoryBound(to: Float.self)
        let ttPtr = tEn.dataPointer.assumingMemoryBound(to: Float.self)

        var outIdx = 0
        for i in 0..<T {
            for _ in 0..<predDur[i] {
                guard outIdx < bucket else { break }
                for c in 0..<dHidden {
                    enPtr[c * bucket + outIdx] = dPtr[c * dStrides[1] + i * dStrides[2]]
                }
                for c in 0..<tHidden {
                    asrPtr[c * bucket + outIdx] = ttPtr[c * tStrides[1] + i * tStrides[2]]
                }
                outIdx += 1
            }
        }

        // Decoder
        let refSArr = try MLMultiArray(shape: [1, 256], dataType: .float32)
        let refSPtr = refSArr.dataPointer.assumingMemoryBound(to: Float.self)
        for (i, v) in refS.enumerated() where i < 256 { refSPtr[i] = v }

        let decOutput = try await decoder.prediction(from:
            MLDictionaryFeatureProvider(dictionary: [
                "en_aligned": MLFeatureValue(multiArray: enAligned),
                "asr_aligned": MLFeatureValue(multiArray: asrAligned),
                "ref_s": MLFeatureValue(multiArray: refSArr)
            ]))
        guard let audioArr = decOutput.featureNames.compactMap({
            decOutput.featureValue(for: $0)?.multiArrayValue
        }).first else {
            throw CMZError.inferenceFailed(reason: "Kokoro decoder output missing")
        }

        // Trim to the frame-accurate duration (600 samples/frame @ 24 kHz).
        let trimLen = min(totalFrames * 600, audioArr.count)
        var audio = [Float](repeating: 0, count: trimLen)
        if audioArr.dataType == .float16 {
            let fp16 = audioArr.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<trimLen { audio[i] = Float(fp16[i]) }
        } else {
            let fp32 = audioArr.dataPointer.assumingMemoryBound(to: Float.self)
            memcpy(&audio, fp32, trimLen * 4)
        }
        return audio
    }

    // MARK: - Resources

    private static func loadNamed(modelId: String, substring: String,
                                   compute: MLComputeUnits) async throws -> MLModel {
        try await ModelLoading.loadSubmodel(modelId: modelId,
                                             containing: substring,
                                             compute: compute,
                                             missingLabel: "Kokoro sub-model")
    }

    private static func loadVocab(modelId: String) throws -> [String: Int32] {
        let dir = CMZPaths.modelDir(id: modelId)
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)) ?? []
        guard let vocabURL = entries.first(where: {
            $0.pathExtension == "json" && $0.lastPathComponent.lowercased().contains("vocab")
        }) ?? entries.first(where: { $0.pathExtension == "json" }) else {
            throw CMZError.inferenceFailed(reason: "kokoro_vocab.json missing")
        }
        let data = try Data(contentsOf: vocabURL)
        guard let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CMZError.inferenceFailed(reason: "vocab JSON shape")
        }
        var vocab: [String: Int32] = [:]
        for (k, v) in raw {
            if let id = v as? Int { vocab[k] = Int32(id) }
            else if let piece = v as? String, let id = Int32(k) { vocab[piece] = id }
        }
        return vocab
    }

    /// Kokoro voice files are `voice_<id>.bin`, containing a 512×256 float32
    /// table. The row indexed by `min(tokenCount − 1, 509)` gives the 256-d
    /// style embedding used for inference.
    private static func loadVoice(modelId: String, id: String, tokenCount: Int) throws -> [Float] {
        let dir = CMZPaths.modelDir(id: modelId)
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)) ?? []
        guard let url = entries.first(where: {
            $0.pathExtension == "bin" && $0.lastPathComponent.contains(id)
        }) else {
            throw CMZError.inferenceFailed(reason: "voice_\(id).bin missing")
        }
        let data = try Data(contentsOf: url)
        let floats = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let row = max(0, min(tokenCount - 1, 509))
        let start = row * 256
        guard floats.count >= start + 256 else {
            throw CMZError.inferenceFailed(reason: "voice file too short")
        }
        return Array(floats[start..<(start + 256)])
    }
}
