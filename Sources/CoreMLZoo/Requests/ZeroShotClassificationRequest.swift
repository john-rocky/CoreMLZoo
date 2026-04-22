import Foundation
import CoreML
import CoreGraphics
import Accelerate

/// Open-vocabulary image classification via image/text embedding models
/// (SigLIP). Caller supplies candidate captions at inference time.
///
/// Pipeline:
/// 1. Run image encoder on 224×224 BGRA → image embedding
/// 2. For each class string, apply `promptTemplate` (default "a photo of a {}"),
///    tokenize (SentencePiece greedy longest match), run text encoder → text embedding
/// 3. Score = cosine(image, text) × `logitScale` (default 100)
/// 4. Softmax across classes → probabilities
///
/// SigLIP normalizes with (0.5, 0.5), NOT ImageNet — the converted
/// mlpackage bakes this in, so we just feed BGRA bytes.
public struct ZeroShotClassificationRequest: CMZRequest {

    public enum Model: String, Sendable, CaseIterable {
        case siglipBase = "siglip"

        var inputSize: Int { 224 }
    }

    public struct ClassifiedLabel: Sendable {
        public let label: String
        public let score: Float  // softmax probability in [0, 1]
    }

    public struct Input: Sendable {
        public var image: CGImage
        public var candidates: [String]
        public init(image: CGImage, candidates: [String]) {
            self.image = image; self.candidates = candidates
        }
    }

    public let model: Model
    public var computeUnits: CMZComputeUnits
    public var promptTemplate: String = "a photo of a {}"
    public var logitScale: Float = 100

    public init(model: Model = .siglipBase, computeUnits: CMZComputeUnits = .auto) {
        self.model = model; self.computeUnits = computeUnits
    }

    public var modelId: String { model.rawValue }

    public func perform(on input: Input) async throws -> [ClassifiedLabel] {
        let classes = input.candidates.filter { !$0.isEmpty }
        guard !classes.isEmpty else { return [] }

        let compute = (computeUnits == .auto) ? MLComputeUnits.all
                                              : computeUnits.mlComputeUnits
        let imageEnc = try await Self.load(modelId: modelId, containing: "image", compute: compute)
        let textEnc  = try await Self.load(modelId: modelId, containing: "text", compute: compute)
        let vocab = try Self.loadVocab(modelId: modelId)

        // 1. Image embedding
        let size = model.inputSize
        let buffer = try ImageBuffer.bgraBuffer(from: input.image,
                                                size: CGSize(width: size, height: size))
        let imgInputKey = imageEnc.modelDescription.inputDescriptionsByName.first {
            $0.value.type == .image
        }?.key ?? "image"
        let imgOutput = try await imageEnc.prediction(from:
            MLDictionaryFeatureProvider(dictionary: [
                imgInputKey: MLFeatureValue(pixelBuffer: buffer)
            ]))
        let imgEmbName = imgOutput.featureNames.first ?? "image_embedding"
        guard let imgEmb = imgOutput.featureValue(for: imgEmbName)?.multiArrayValue else {
            throw CMZError.inferenceFailed(reason: "image embedding missing")
        }
        let imageVec = Self.readFloats(imgEmb)

        // 2-3. Text embeddings + cosine scoring
        let txtInputKey = textEnc.modelDescription.inputDescriptionsByName.keys.first {
            $0.contains("input")
        } ?? "input_ids"

        var logits: [(String, Float)] = []
        for cls in classes {
            let prompt = promptTemplate.replacingOccurrences(of: "{}", with: cls)
            let tokens = Self.sentencePieceTokenize(prompt, vocab: vocab)
            let tokenArr = try MLMultiArray(shape: [1, NSNumber(value: tokens.count)],
                                            dataType: .int32)
            for (i, t) in tokens.enumerated() { tokenArr[i] = NSNumber(value: t) }
            let txtOutput = try await textEnc.prediction(from:
                MLDictionaryFeatureProvider(dictionary: [
                    txtInputKey: MLFeatureValue(multiArray: tokenArr)
                ]))
            let txtEmbName = txtOutput.featureNames.first ?? "text_embedding"
            guard let txtEmb = txtOutput.featureValue(for: txtEmbName)?.multiArrayValue else {
                continue
            }
            let textVec = Self.readFloats(txtEmb)
            logits.append((cls, Self.cosine(imageVec, textVec) * logitScale))
        }

        // 4. Softmax
        let maxL = logits.map(\.1).max() ?? 0
        let exps = logits.map { exp($0.1 - maxL) }
        let sum = exps.reduce(0, +)
        let probs: [ClassifiedLabel] = zip(logits, exps).map { pair, e in
            ClassifiedLabel(label: pair.0, score: sum > 0 ? e / sum : 0)
        }
        return probs.sorted { $0.score > $1.score }
    }

    // MARK: - SentencePiece tokenize (greedy longest match, SigLIP convention)

    private static func sentencePieceTokenize(_ text: String,
                                                vocab: [String: Int]) -> [Int32] {
        let processed = "\u{2581}" + text.lowercased()
            .replacingOccurrences(of: " ", with: "\u{2581}")
        var tokens: [Int32] = []
        var i = processed.startIndex
        while i < processed.endIndex {
            var bestLen = 0
            var bestId: Int32 = 0
            let maxLen = min(20, processed.distance(from: i, to: processed.endIndex))
            for len in (1...maxLen).reversed() {
                let end = processed.index(i, offsetBy: len)
                let piece = String(processed[i..<end])
                if let id = vocab[piece] { bestLen = len; bestId = Int32(id); break }
            }
            if bestLen == 0 { i = processed.index(after: i) }
            else { tokens.append(bestId); i = processed.index(i, offsetBy: bestLen) }
        }
        if let eos = vocab["</s>"] { tokens.append(Int32(eos)) }
        return tokens
    }

    // MARK: - Helpers

    private static func load(modelId: String, containing needle: String,
                              compute: MLComputeUnits) async throws -> MLModel {
        try await ModelLoading.loadSubmodel(modelId: modelId,
                                             containing: needle,
                                             compute: compute,
                                             missingLabel: "sub-model containing")
    }

    private static func loadVocab(modelId: String) throws -> [String: Int] {
        let dir = CMZPaths.modelDir(id: modelId)
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)) ?? []
        guard let vocabURL = entries.first(where: {
            $0.pathExtension == "json" && $0.lastPathComponent.lowercased().contains("siglip_vocab")
        }) ?? entries.first(where: { $0.pathExtension == "json" }) else {
            throw CMZError.inferenceFailed(reason: "siglip_vocab.json missing")
        }
        let data = try Data(contentsOf: vocabURL)
        guard let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CMZError.inferenceFailed(reason: "vocab JSON shape")
        }
        var vocab: [String: Int] = [:]
        for (k, v) in raw {
            if let id = v as? Int { vocab[k] = id }
            else if let piece = v as? String, let id = Int(k) { vocab[piece] = id }
        }
        return vocab
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

    private static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        let n = min(a.count, b.count)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<n {
            dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
        }
        return dot / max(1e-8, (na.squareRoot() * nb.squareRoot()))
    }
}
