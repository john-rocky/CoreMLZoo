import Foundation
import CoreML
import CoreGraphics

/// Florence-2 multitask vision-language session.
///
/// Florence-2 is distributed as three mlpackages (vision encoder + text
/// encoder + decoder, INT8, ~260 MB total) plus a BART/GPT-2-style vocab
/// JSON. The pipeline is:
///
/// 1. `vision encoder`: image → image_features
/// 2. `text encoder`: image_features + task prompt tokens → encoder_hidden_states
/// 3. `decoder`: autoregressively decodes tokens (no KV cache — each step
///    re-runs the full sequence)
///
/// All three mlpackages plus `florence2_vocab.json` are expected under the
/// single manifest entry `florence2_base`. Compute units default to
/// `.cpuOnly` because the encoders are ≥ 768×768 ViTs that hit ANE E5
/// buffer limits.
///
/// ```swift
/// let session = try await VisionLanguageSession(model: .florence2Base)
/// let caption = try await session.caption(cgImage)                   // "<CAPTION>"
/// let detail  = try await session.caption(cgImage, detail: .detailed)
/// let text    = try await session.ocr(cgImage)
/// let answer  = try await session.ask("What color is the car?", in: cgImage)
/// ```
public final class VisionLanguageSession: CMZSession {

    public enum Model: Sendable {
        case florence2
        var ids: [String] {
            switch self { case .florence2: return ["florence2"] }
        }
        fileprivate var imageSize: Int { 768 }
    }

    public enum CaptionDetail: Sendable { case brief, detailed, moreDetailed }

    public let model: Model
    public var modelIds: [String] { model.ids }

    /// Token-to-string map keyed by token id.
    private let reverseVocab: [Int: String]
    /// String-to-id map for BPE merging.
    private let forwardVocab: [String: Int]

    private let visionEncoder: MLModel
    private let textEncoder: MLModel
    private let decoder: MLModel

    /// BART/Florence-2 special tokens.
    private let bosTokenId: Int32 = 0
    private let eosTokenId: Int32 = 2
    private let decoderStartTokenId: Int32 = 2
    private let specialTokenIds: Set<Int32> = [0, 1, 2]

    public init(model: Model = .florence2,
                computeUnits: CMZComputeUnits = .auto) async throws {
        self.model = model
        let modelId = model.ids[0]
        let compute = (computeUnits == .auto) ? MLComputeUnits.cpuOnly
                                              : computeUnits.mlComputeUnits

        // Locate the three sub-packages inside the single florence2_base
        // install directory by name substring (hub app convention).
        async let ve = Self.loadNamed(modelId: modelId, containing: "VisionEncoder", compute: compute)
        async let te = Self.loadNamed(modelId: modelId, containing: "TextEncoder",   compute: compute)
        async let dc = Self.loadNamed(modelId: modelId, containing: "Decoder",       compute: compute)
        self.visionEncoder = try await ve
        self.textEncoder   = try await te
        self.decoder       = try await dc

        // Load vocab JSON next to the mlpackages.
        let (rv, fv) = try Self.loadVocab(modelId: modelId)
        self.reverseVocab = rv
        self.forwardVocab = fv
    }

    // MARK: - Public tasks

    public func caption(_ image: CGImage,
                        detail: CaptionDetail = .brief,
                        maxTokens: Int = 256) async throws -> String {
        let taskTokens: [Int32] = {
            switch detail {
            case .brief:        return [0, 2264, 473, 5, 2274, 6190, 116, 2]
            case .detailed:     return [0, 47066, 21700, 11, 4617, 99, 16, 2343, 11, 5, 2274, 4, 2]
            case .moreDetailed: return [0, 47066, 21700, 19, 10, 17818, 99, 16, 2343, 11, 5, 2274, 4, 2]
            }
        }()
        return try await infer(image: image, promptTokens: taskTokens, maxTokens: maxTokens)
    }

    public func ocr(_ image: CGImage, maxTokens: Int = 512) async throws -> String {
        let tokens: [Int32] = [0, 2264, 16, 5, 2788, 11, 5, 2274, 116, 2]
        return try await infer(image: image, promptTokens: tokens, maxTokens: maxTokens)
    }

    public func ask(_ question: String, in image: CGImage, maxTokens: Int = 256) async throws -> String {
        let tokens = tokenize(question)
        return try await infer(image: image, promptTokens: tokens, maxTokens: maxTokens)
    }

    /// Phrase grounding. Returns normalized bounding boxes (origin top-left,
    /// [0, 1]^4, Vision convention) for every region in `image` that matches
    /// `phrase`.
    ///
    /// Florence-2 encodes grounding output as `<phrase><loc_X1><loc_Y1><loc_X2><loc_Y2>`
    /// with each `loc_N` being a quantized coordinate in `0..<1000`. This
    /// method parses those markers out of the decoded string and returns
    /// the normalized rects.
    public func groundPhrase(_ phrase: String,
                             in image: CGImage,
                             maxTokens: Int = 256) async throws -> [CGRect] {
        let prompt = "Locate the phrases in the caption: \(phrase)"
        let tokens = tokenize(prompt)
        let raw = try await infer(image: image,
                                   promptTokens: tokens,
                                   maxTokens: maxTokens)
        return Self.parseLocationBoxes(from: raw)
    }

    /// Parse `<loc_X>` markers out of a Florence-2 decoded string into
    /// normalized `[0, 1]^4` rects. Markers come in groups of 4 ordered
    /// (x1, y1, x2, y2); we clamp, reject degenerate rects, and return
    /// Vision-convention rects (origin top-left).
    static func parseLocationBoxes(from raw: String) -> [CGRect] {
        // Extract every integer N appearing inside `<loc_N>`.
        let pattern = "<loc_(\\d+)>"
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let ns = raw as NSString
        let matches = regex.matches(in: raw, range: NSRange(location: 0, length: ns.length))
        var coords: [Int] = []
        coords.reserveCapacity(matches.count)
        for m in matches where m.numberOfRanges >= 2 {
            let digits = ns.substring(with: m.range(at: 1))
            if let v = Int(digits) { coords.append(v) }
        }

        var rects: [CGRect] = []
        let groups = coords.count / 4
        let scale: CGFloat = 1.0 / 999.0
        for g in 0..<groups {
            let base = g * 4
            var x1 = CGFloat(coords[base    ]) * scale
            var y1 = CGFloat(coords[base + 1]) * scale
            var x2 = CGFloat(coords[base + 2]) * scale
            var y2 = CGFloat(coords[base + 3]) * scale
            if x2 < x1 { swap(&x1, &x2) }
            if y2 < y1 { swap(&y1, &y2) }
            x1 = max(0, min(1, x1)); y1 = max(0, min(1, y1))
            x2 = max(0, min(1, x2)); y2 = max(0, min(1, y2))
            let w = x2 - x1, h = y2 - y1
            if w > 0 && h > 0 {
                rects.append(CGRect(x: x1, y: y1, width: w, height: h))
            }
        }
        return rects
    }

    // MARK: - Pipeline

    private func infer(image: CGImage, promptTokens: [Int32], maxTokens: Int) async throws -> String {
        // 1. Vision encode
        let buffer = try ImageBuffer.bgraBuffer(from: image,
            size: CGSize(width: model.imageSize, height: model.imageSize))
        let veOutput = try await visionEncoder.prediction(from:
            MLDictionaryFeatureProvider(dictionary: [
                "image": MLFeatureValue(pixelBuffer: buffer)
            ]))
        guard let rawFeat = veOutput.featureValue(for: "image_features")?.multiArrayValue else {
            throw CMZError.inferenceFailed(reason: "VisionEncoder: image_features missing")
        }
        let imageFeatures = try Self.copyArray(rawFeat)

        // 2. Text encode
        let idsArray = try MLMultiArray(shape: [1, NSNumber(value: promptTokens.count)],
                                        dataType: .int32)
        for (i, t) in promptTokens.enumerated() {
            idsArray[[0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: t)
        }
        let teOutput = try await textEncoder.prediction(from:
            MLDictionaryFeatureProvider(dictionary: [
                "image_features": imageFeatures,
                "input_ids": idsArray
            ]))
        guard let rawHS = teOutput.featureValue(for: "encoder_hidden_states")?.multiArrayValue else {
            throw CMZError.inferenceFailed(reason: "TextEncoder: encoder_hidden_states missing")
        }
        let encoderHiddenStates = try Self.copyArray(rawHS)

        // 3. Decoder loop (argmax greedy)
        var generated: [Int32] = [decoderStartTokenId]
        for _ in 0..<maxTokens {
            let decIds = try MLMultiArray(shape: [1, NSNumber(value: generated.count)],
                                          dataType: .int32)
            for (i, t) in generated.enumerated() {
                decIds[[0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: t)
            }
            let decOutput = try await decoder.prediction(from:
                MLDictionaryFeatureProvider(dictionary: [
                    "decoder_input_ids": decIds,
                    "encoder_hidden_states": encoderHiddenStates
                ]))
            guard let logits = decOutput.featureValue(for: "logits")?.multiArrayValue else { break }
            let next = Self.argmaxLastToken(logits)
            if next == eosTokenId { break }
            generated.append(next)
        }
        return decode(Array(generated.dropFirst()))
    }

    // MARK: - Tokenizer (BART / GPT-2 byte-level BPE)

    /// Pre-tokenize with GPT-2's regex, byte-encode, greedy-longest match
    /// against the forward vocab.
    private func tokenize(_ text: String) -> [Int32] {
        var tokens: [Int32] = [bosTokenId]
        let pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
        let regex = try! NSRegularExpression(pattern: pattern, options: [])
        let ns = text as NSString
        for m in regex.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            let piece = ns.substring(with: m.range)
            let encoded = byteEncode(piece)
            var i = encoded.startIndex
            while i < encoded.endIndex {
                var bestEnd = encoded.index(after: i)
                var bestId: Int?
                var j = encoded.index(after: i)
                while j <= encoded.endIndex {
                    let sub = String(encoded[i..<j])
                    if let id = forwardVocab[sub] { bestEnd = j; bestId = id }
                    if j == encoded.endIndex { break }
                    j = encoded.index(after: j)
                }
                if let id = bestId { tokens.append(Int32(id)) }
                i = bestEnd
            }
        }
        tokens.append(eosTokenId)
        return tokens
    }

    private func byteEncode(_ text: String) -> String {
        var out = ""
        for b in Array(text.utf8) {
            if let ch = Self.byteEncoder[b] { out.append(ch) }
        }
        return out
    }

    private func decode(_ tokenIds: [Int32]) -> String {
        var pieces: [String] = []
        for id in tokenIds {
            if specialTokenIds.contains(id) { continue }
            if let piece = reverseVocab[Int(id)] { pieces.append(piece) }
        }
        return pieces.joined()
            .replacingOccurrences(of: "\u{0120}", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Byte → unicode map used by GPT-2/RoBERTa/BART tokenizers so that
    /// arbitrary bytes become printable Unicode before BPE lookup.
    private static let byteEncoder: [UInt8: Character] = {
        var direct = Set<UInt8>()
        for b in 0x21...0x7E { direct.insert(UInt8(b)) }
        for b in 0xA1...0xAC { direct.insert(UInt8(b)) }
        for b in 0xAE...0xFF { direct.insert(UInt8(b)) }
        var map: [UInt8: Character] = [:]
        for b in direct { map[b] = Character(UnicodeScalar(b)) }
        var n = 0
        for b: UInt8 in 0...255 {
            if !direct.contains(b) { map[b] = Character(UnicodeScalar(256 + n)!); n += 1 }
        }
        return map
    }()

    // MARK: - Model / vocab loading

    private static func loadNamed(modelId: String, containing needle: String,
                                   compute: MLComputeUnits) async throws -> MLModel {
        let dir = CMZPaths.modelDir(id: modelId)
        let fm = FileManager.default
        guard fm.fileExists(atPath: CMZPaths.metaFile(modelId: modelId).path) else {
            throw CMZError.modelNotInstalled(id: modelId)
        }
        let entries = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        guard let url = entries.first(where: {
            ($0.pathExtension == "mlpackage" || $0.pathExtension == "mlmodelc")
                && $0.lastPathComponent.contains(needle)
        }) else {
            throw CMZError.inferenceFailed(reason: "Florence-2 sub-model '\(needle)' missing")
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = compute
        do {
            return try await ModelLoading.loadCompiled(at: url, configuration: cfg)
        } catch {
            throw CMZError.inferenceFailed(reason: "load \(url.lastPathComponent): \(error)")
        }
    }

    private static func loadVocab(modelId: String) throws -> ([Int: String], [String: Int]) {
        let dir = CMZPaths.modelDir(id: modelId)
        let fm = FileManager.default
        let entries = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        guard let vocabURL = entries.first(where: {
            $0.pathExtension == "json" && $0.lastPathComponent.lowercased().contains("vocab")
        }) else {
            throw CMZError.inferenceFailed(reason: "florence2_vocab.json missing alongside mlpackages")
        }
        let data = try Data(contentsOf: vocabURL)
        guard let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CMZError.inferenceFailed(reason: "vocab JSON is not a dict")
        }
        var reverse: [Int: String] = [:], forward: [String: Int] = [:]
        for (k, v) in raw {
            // Hub App tolerates both {id_string: piece} and {piece: id} shapes.
            if let id = Int(k), let piece = v as? String {
                reverse[id] = piece; forward[piece] = id
            } else if let id = v as? Int {
                reverse[id] = k; forward[k] = id
            }
        }
        return (reverse, forward)
    }

    private static func copyArray(_ src: MLMultiArray) throws -> MLMultiArray {
        let dst = try MLMultiArray(shape: src.shape, dataType: src.dataType)
        let bytes: Int
        switch src.dataType {
        case .float16:           bytes = src.count * 2
        case .float32, .int32:   bytes = src.count * 4
        case .float64:           bytes = src.count * 8
        default:                 bytes = src.count * 4
        }
        memcpy(dst.dataPointer, src.dataPointer, bytes)
        return dst
    }

    private static func argmaxLastToken(_ logits: MLMultiArray) -> Int32 {
        let vocabSize = logits.shape.last?.intValue ?? 0
        let offset = logits.count - vocabSize
        if logits.dataType == .float16 {
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float16.self)
            var idx = 0, best = ptr[offset]
            for i in 1..<vocabSize {
                let v = ptr[offset + i]; if v > best { best = v; idx = i }
            }
            return Int32(idx)
        } else {
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float32.self)
            var idx = 0, best = ptr[offset]
            for i in 1..<vocabSize {
                let v = ptr[offset + i]; if v > best { best = v; idx = i }
            }
            return Int32(idx)
        }
    }
}
