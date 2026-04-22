import Foundation
import CoreML

/// Text → music generation. Default: Stable Audio Open Small
/// (T5 encoder + Number embedder + DiT + VAE decoder).
///
/// Pipeline:
/// 1. SentencePiece T5 tokenize prompt → [1, 64] int32 ids
/// 2. T5 encoder → [1, 64, 768] text embeddings (NaN-sanitised)
/// 3. Number embedder: normalized seconds → [1, 768]
/// 4. Build `cross_attn_cond [1, 65, 768]` (text || seconds) + `global_embed [1, 768]`
/// 5. Init FP16 noise latent [1, 64, 256] via LCG seeded Box-Muller
/// 6. Denoise loop: logit-SNR schedule, Euler step with DiT velocity
/// 7. VAE decode → stereo waveform, trim to requested duration
///
/// DiT inference uses FP16 by default; on iPad Pro / Mac, attention
/// overflow can cause audible artifacts — override `computeUnits` with
/// `.cpuOnly` as a mitigation.
public final class TextToMusicSession: CMZSession {

    public enum Model: Sendable {
        case stableAudioOpenSmall
        var id: String { "stable_audio" }
        var ditSubstring: String { "DiT" }
    }

    public struct Result: Sendable {
        /// Stereo-interleaved 44.1 kHz float PCM (left[0], right[0], left[1], …).
        public let waveform: [Float]
        public let sampleRate: Int
        public let durationSeconds: Double
    }

    public let model: Model
    public var modelIds: [String] { [model.id] }

    public static let sampleRate = 44100
    static let tokenLen = 64
    static let embDim = 768
    static let latentFrames = 64
    static let latentChannels = 256
    static let maxAudioSamples = 524288

    private let t5Encoder: MLModel
    private let numberEmbedder: MLModel
    private let dit: MLModel
    private let vaeDecoder: MLModel
    private let vocab: [String: Int32]

    public init(model: Model = .stableAudioOpenSmall,
                computeUnits: CMZComputeUnits = .auto) async throws {
        self.model = model
        let modelId = model.id
        let compute = (computeUnits == .auto) ? MLComputeUnits.all
                                              : computeUnits.mlComputeUnits

        async let t5 = Self.load(modelId: modelId, substring: "T5Encoder", exclude: nil, compute: compute)
        async let ne = Self.load(modelId: modelId, substring: "NumberEmbedder", exclude: nil, compute: compute)
        async let di = Self.load(modelId: modelId, substring: "DiT", exclude: nil, compute: compute)
        async let va = Self.load(modelId: modelId, substring: "VAEDecoder", exclude: nil, compute: compute)
        self.t5Encoder = try await t5
        self.numberEmbedder = try await ne
        self.dit = try await di
        self.vaeDecoder = try await va
        self.vocab = try Self.loadVocab(modelId: modelId)
    }

    // MARK: - Public API

    public func generate(prompt: String,
                         durationSeconds: Double = 10,
                         seed: UInt64? = nil,
                         steps: Int = 25) async throws -> Result {
        let duration = min(max(0.5, durationSeconds), 11.0)
        let seedValue = seed ?? UInt64.random(in: 0...UInt64.max)

        // 1. Tokenize
        let tokens = Self.t5Tokenize(prompt, vocab: vocab, maxLen: Self.tokenLen)

        // 2. T5 encode
        let inputIds = try MLMultiArray(shape: [1, NSNumber(value: Self.tokenLen)],
                                         dataType: .int32)
        let idPtr = inputIds.dataPointer.assumingMemoryBound(to: Int32.self)
        for (i, t) in tokens.enumerated() { idPtr[i] = t }

        let t5Out = try await t5Encoder.prediction(from:
            MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: inputIds)
            ]))
        guard let textEmb = t5Out.featureValue(for: "text_embeddings")?.multiArrayValue else {
            throw CMZError.inferenceFailed(reason: "T5 text_embeddings missing")
        }

        // Sanitise NaN/Inf from INT8 quant + zero out padding after the first EOS.
        let attnLen = tokens.firstIndex(of: 0) ?? Self.tokenLen
        for t in 0..<Self.tokenLen {
            for c in 0..<Self.embDim {
                let idx = [0, t, c] as [NSNumber]
                let v = textEmb[idx].floatValue
                if t >= attnLen || v.isNaN || v.isInfinite {
                    textEmb[idx] = 0
                }
            }
        }

        // 3. Number embedder
        let normSec = try MLMultiArray(shape: [1], dataType: .float16)
        normSec.dataPointer.assumingMemoryBound(to: Float16.self)[0]
            = Float16(min(max(Float(duration), 0), 256) / 256.0)
        let neOut = try await numberEmbedder.prediction(from:
            MLDictionaryFeatureProvider(dictionary: [
                "normalized_seconds": MLFeatureValue(multiArray: normSec)
            ]))
        guard let secEmb = neOut.featureValue(for: "seconds_embedding")?.multiArrayValue else {
            throw CMZError.inferenceFailed(reason: "seconds_embedding missing")
        }

        // 4. cross_attn_cond + global_embed
        let crossAttn = try MLMultiArray(shape: [1, 65, NSNumber(value: Self.embDim)],
                                          dataType: .float16)
        let globalEmb = try MLMultiArray(shape: [1, NSNumber(value: Self.embDim)],
                                          dataType: .float16)
        let caPtr = crossAttn.dataPointer.assumingMemoryBound(to: Float16.self)
        let gePtr = globalEmb.dataPointer.assumingMemoryBound(to: Float16.self)
        for t in 0..<Self.tokenLen {
            for c in 0..<Self.embDim {
                caPtr[t * Self.embDim + c]
                    = Float16(textEmb[[0, t, c] as [NSNumber]].floatValue)
            }
        }
        for c in 0..<Self.embDim {
            let v = secEmb[c].floatValue
            caPtr[64 * Self.embDim + c] = Float16(v)
            gePtr[c] = Float16(v)
        }

        // 5. Noise latent [1, 64, 256], Box-Muller seeded by LCG
        let latent = try MLMultiArray(shape: [1, NSNumber(value: Self.latentFrames),
                                                NSNumber(value: Self.latentChannels)],
                                       dataType: .float16)
        let lPtr = latent.dataPointer.assumingMemoryBound(to: Float16.self)
        let count = Self.latentFrames * Self.latentChannels
        var rng: UInt64 = seedValue
        for i in stride(from: 0, to: count, by: 2) {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let u1 = max(Float.ulpOfOne, Float(rng >> 33) / Float(1 << 31))
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let u2 = Float(rng >> 33) / Float(1 << 31)
            let r = sqrt(-2 * log(u1))
            lPtr[i] = Float16(r * cos(2 * .pi * u2))
            if i + 1 < count { lPtr[i + 1] = Float16(r * sin(2 * .pi * u2)) }
        }

        // 6. Denoise loop (Euler + logit-SNR schedule)
        let schedule = Self.logitSNRSchedule(steps: steps)
        for i in 0..<steps {
            let tCurr = schedule[i], tNext = schedule[i + 1]
            let dt = tNext - tCurr
            let ts = try MLMultiArray(shape: [1], dataType: .float16)
            ts.dataPointer.assumingMemoryBound(to: Float16.self)[0] = Float16(tCurr)

            let ditOut = try await dit.prediction(from:
                MLDictionaryFeatureProvider(dictionary: [
                    "latent": MLFeatureValue(multiArray: latent),
                    "timestep": MLFeatureValue(multiArray: ts),
                    "cross_attn_cond": MLFeatureValue(multiArray: crossAttn),
                    "global_embed": MLFeatureValue(multiArray: globalEmb)
                ]))
            guard let velocity = ditOut.featureValue(for: "velocity")?.multiArrayValue else {
                continue
            }
            let vPtr = velocity.dataPointer
            if velocity.dataType == .float16 {
                let vfp16 = vPtr.assumingMemoryBound(to: Float16.self)
                for j in 0..<count {
                    lPtr[j] = Float16(Float(lPtr[j]) + dt * Float(vfp16[j]))
                }
            } else {
                let vfp32 = vPtr.assumingMemoryBound(to: Float.self)
                for j in 0..<count {
                    lPtr[j] = Float16(Float(lPtr[j]) + dt * vfp32[j])
                }
            }
        }

        // 7. VAE decode
        let vaeOut = try await vaeDecoder.prediction(from:
            MLDictionaryFeatureProvider(dictionary: [
                "latent": MLFeatureValue(multiArray: latent)
            ]))
        guard let audio = vaeOut.featureValue(for: "audio")?.multiArrayValue else {
            throw CMZError.inferenceFailed(reason: "VAE audio output missing")
        }
        let sr = Self.sampleRate
        let trim = min(Int(duration * Double(sr)), Self.maxAudioSamples)
        var stereoInterleaved = [Float](repeating: 0, count: trim * 2)
        for i in 0..<trim {
            let l = audio[[0, 0, i] as [NSNumber]].floatValue
            let r = audio[[0, 1, i] as [NSNumber]].floatValue
            stereoInterleaved[i * 2]     = l
            stereoInterleaved[i * 2 + 1] = r
        }
        return Result(waveform: stereoInterleaved,
                      sampleRate: sr,
                      durationSeconds: Double(trim) / Double(sr))
    }

    // MARK: - Schedule + Tokenizer

    private static func logitSNRSchedule(steps: Int) -> [Float] {
        var s = [Float](repeating: 0, count: steps + 1)
        for i in 0...steps {
            let logsnr: Float = -6.0 + Float(i) / Float(steps) * 8.0
            s[i] = 1.0 / (1.0 + exp(logsnr))
        }
        s[0] = 1.0; s[steps] = 0.0
        return s
    }

    private static func t5Tokenize(_ text: String,
                                    vocab: [String: Int32],
                                    maxLen: Int) -> [Int32] {
        let processed = "\u{2581}" + text.lowercased()
            .replacingOccurrences(of: " ", with: "\u{2581}")
        var tokens: [Int32] = []
        var i = processed.startIndex
        while i < processed.endIndex {
            var bestLen = 0, bestId: Int32 = 0
            let maxPiece = min(20, processed.distance(from: i, to: processed.endIndex))
            for len in (1...maxPiece).reversed() {
                let end = processed.index(i, offsetBy: len)
                if let id = vocab[String(processed[i..<end])] {
                    bestLen = len; bestId = id; break
                }
            }
            if bestLen == 0 { i = processed.index(after: i) }
            else { tokens.append(bestId); i = processed.index(i, offsetBy: bestLen) }
        }
        tokens.append(1)  // T5 EOS
        if tokens.count > maxLen { tokens = Array(tokens.prefix(maxLen - 1)) + [1] }
        var result = [Int32](repeating: 0, count: maxLen)
        for (idx, t) in tokens.enumerated() { result[idx] = t }
        return result
    }

    // MARK: - Resource loading

    private static func load(modelId: String, substring: String,
                              exclude: String?,
                              compute: MLComputeUnits) async throws -> MLModel {
        try await ModelLoading.loadSubmodel(modelId: modelId,
                                             containing: substring,
                                             excluding: exclude,
                                             caseInsensitive: false,
                                             compute: compute,
                                             missingLabel: "Stable Audio")
    }

    private static func loadVocab(modelId: String) throws -> [String: Int32] {
        let dir = CMZPaths.modelDir(id: modelId)
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)) ?? []
        guard let vocabURL = entries.first(where: {
            $0.pathExtension == "json" && $0.lastPathComponent.lowercased().contains("t5")
        }) ?? entries.first(where: { $0.pathExtension == "json" }) else {
            throw CMZError.inferenceFailed(reason: "t5_vocab.json missing")
        }
        let data = try Data(contentsOf: vocabURL)
        guard let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CMZError.inferenceFailed(reason: "t5_vocab JSON shape")
        }
        var vocab: [String: Int32] = [:]
        for (k, v) in raw {
            if let piece = v as? String, let id = Int32(k) { vocab[piece] = id }
            else if let id = v as? Int { vocab[k] = Int32(id) }
        }
        return vocab
    }
}
