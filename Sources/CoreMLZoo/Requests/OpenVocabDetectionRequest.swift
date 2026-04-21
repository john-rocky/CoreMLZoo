import Foundation
import CoreML
import CoreGraphics
import Accelerate

public struct DetectedObject: Sendable {
    public let label: String
    public let confidence: Float
    /// Normalized [0,1]^4, origin top-left (Vision convention).
    public let boundingBox: CGRect
}

/// Open-vocabulary object detection — caller supplies class labels at
/// inference time. Model: YOLO-World V2-S (detector + CLIP ViT-B/32 text
/// encoder + BPE vocab).
///
/// Pipeline:
/// 1. CLIP-tokenize each class string → `text_tokens`
/// 2. CLIP text encoder → per-class 512d embedding, L2-normalized
/// 3. Pack up to 80 embeddings into `txt_feats [1, 80, 512]`
/// 4. Letterbox the image to 640×640 (0.5 gray padding), build
///    `image [1, 3, 640, 640]` in [0, 1]
/// 5. Detector → `boxes [4, numAnchors]`, `scores [numClasses, numAnchors]`
/// 6. Decode + per-class NMS (IoU 0.5)
public struct OpenVocabDetectionRequest: CMZRequest {

    public enum Model: String, Sendable, CaseIterable {
        case yoloWorldV2S = "yoloworld"

        var inputSize: Int { 640 }
        var maxClasses: Int { 80 }
        var embedDim: Int { 512 }
    }

    public struct Input: Sendable {
        public var image: CGImage
        public var classes: [String]
        public var confidenceThreshold: Float
        public var iouThreshold: Float
        public init(image: CGImage, classes: [String],
                    confidenceThreshold: Float = 0.15,
                    iouThreshold: Float = 0.5) {
            self.image = image; self.classes = classes
            self.confidenceThreshold = confidenceThreshold
            self.iouThreshold = iouThreshold
        }
    }

    public let model: Model
    public var computeUnits: CMZComputeUnits

    public init(model: Model = .yoloWorldV2S, computeUnits: CMZComputeUnits = .auto) {
        self.model = model; self.computeUnits = computeUnits
    }

    public var modelId: String { model.rawValue }

    public func perform(on input: Input) async throws -> [DetectedObject] {
        let classes = input.classes.filter { !$0.isEmpty }
        guard !classes.isEmpty else { return [] }

        let compute = (computeUnits == .auto) ? MLComputeUnits.all
                                              : computeUnits.mlComputeUnits

        let detector = try await Self.load(modelId: modelId, containing: "detector", compute: compute)
        let textEnc  = try await Self.load(modelId: modelId, containing: "text_encoder", compute: compute)
        let tokenizer = try Self.loadTokenizer(modelId: modelId)

        // 1-3. Text embeddings → txt_feats
        let txtFeats = try Self.encodeTextQueries(classes,
                                                   tokenizer: tokenizer,
                                                   textEncoder: textEnc,
                                                   maxClasses: model.maxClasses,
                                                   embedDim: model.embedDim)

        // 4. Letterbox image
        let inputSize = model.inputSize
        let imgW = input.image.width, imgH = input.image.height
        let scale = Float(inputSize) / Float(max(imgW, imgH))
        let scaledW = Int(Float(imgW) * scale)
        let scaledH = Int(Float(imgH) * scale)
        let padX = (inputSize - scaledW) / 2
        let padY = (inputSize - scaledH) / 2
        let imageTensor = try Self.letterbox(input.image,
                                              size: inputSize,
                                              scaledW: scaledW, scaledH: scaledH,
                                              padX: padX, padY: padY)

        // 5. Detector
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "image": imageTensor, "txt_feats": txtFeats
        ])
        let output: MLFeatureProvider
        do {
            output = try await detector.prediction(from: provider)
        } catch {
            throw CMZError.inferenceFailed(reason: "detector: \(error)")
        }
        guard let boxesMA = output.featureValue(for: "boxes")?.multiArrayValue,
              let scoresMA = output.featureValue(for: "scores")?.multiArrayValue else {
            throw CMZError.inferenceFailed(reason: "missing boxes/scores")
        }

        let boxes = Self.readFloats(boxesMA)
        let scores = Self.readFloats(scoresMA)
        let scShape = scoresMA.shape.map { $0.intValue }
        let numClasses = scShape.count >= 2 ? scShape[1] : model.maxClasses
        let numAnchors = scShape.count >= 3 ? scShape[2] : (scores.count / max(1, numClasses))

        // 6. Decode + NMS
        var all: [(CGRect, Float, Int)] = []
        for qi in 0..<min(classes.count, numClasses) {
            let off = qi * numAnchors
            for a in 0..<numAnchors {
                let s = scores[off + a]
                if s < input.confidenceThreshold { continue }
                let cx = boxes[0 * numAnchors + a]
                let cy = boxes[1 * numAnchors + a]
                let bw = boxes[2 * numAnchors + a]
                let bh = boxes[3 * numAnchors + a]

                let nx = (cx - bw / 2 - Float(padX)) / (Float(imgW) * scale)
                let ny = (cy - bh / 2 - Float(padY)) / (Float(imgH) * scale)
                let nw = bw / (Float(imgW) * scale)
                let nh = bh / (Float(imgH) * scale)
                let rect = CGRect(
                    x: CGFloat(max(0, min(1, nx))), y: CGFloat(max(0, min(1, ny))),
                    width: CGFloat(max(0, min(1, nw))), height: CGFloat(max(0, min(1, nh)))
                )
                all.append((rect, s, qi))
            }
        }
        all.sort { $0.1 > $1.1 }
        var kept: [Int] = []
        for i in all.indices {
            var suppress = false
            for k in kept where all[i].2 == all[k].2
                && Self.iou(all[i].0, all[k].0) > input.iouThreshold {
                suppress = true; break
            }
            if !suppress { kept.append(i) }
        }
        return kept.prefix(100).map {
            DetectedObject(label: classes[all[$0].2],
                           confidence: all[$0].1,
                           boundingBox: all[$0].0)
        }
    }

    // MARK: - Internals

    private static func encodeTextQueries(_ queries: [String],
                                           tokenizer: CLIPTokenizer,
                                           textEncoder: MLModel,
                                           maxClasses: Int,
                                           embedDim: Int) throws -> MLMultiArray {
        let arr = try MLMultiArray(
            shape: [1, NSNumber(value: maxClasses), NSNumber(value: embedDim)],
            dataType: .float32)
        memset(arr.dataPointer, 0, maxClasses * embedDim * 4)
        let dst = arr.dataPointer.assumingMemoryBound(to: Float.self)

        let ctxLen = tokenizer.contextLength
        for (i, q) in queries.prefix(maxClasses).enumerated() {
            let tokens = tokenizer.tokenize(q)
            // The detector was traced with batch dimension = maxClasses on
            // the text encoder input; zero-pad the rest of the rows.
            let tokenArr = try MLMultiArray(
                shape: [NSNumber(value: maxClasses), NSNumber(value: ctxLen)],
                dataType: .int32)
            memset(tokenArr.dataPointer, 0, maxClasses * ctxLen * 4)
            let tp = tokenArr.dataPointer.assumingMemoryBound(to: Int32.self)
            for j in 0..<ctxLen { tp[j] = Int32(tokens[j]) }

            let input = try MLDictionaryFeatureProvider(
                dictionary: ["text_tokens": tokenArr])
            let output = try textEncoder.prediction(from: input)
            guard let emb = output.featureValue(for: "text_embeddings")?.multiArrayValue else {
                continue
            }
            var embFloats = readFloats(emb)
            if embFloats.count < embedDim {
                embFloats += [Float](repeating: 0, count: embedDim - embFloats.count)
            } else if embFloats.count > embedDim {
                embFloats = Array(embFloats.prefix(embedDim))
            }
            // L2 normalize
            var sq: Float = 0
            vDSP_svesq(embFloats, 1, &sq, vDSP_Length(embedDim))
            let norm = sqrt(sq)
            if norm > 1e-8 {
                var s: Float = 1 / norm
                vDSP_vsmul(embFloats, 1, &s, &embFloats, 1, vDSP_Length(embedDim))
            }
            for j in 0..<embedDim { dst[i * embedDim + j] = embFloats[j] }
        }
        return arr
    }

    private static func letterbox(_ image: CGImage,
                                   size: Int,
                                   scaledW: Int, scaledH: Int,
                                   padX: Int, padY: Int) throws -> MLMultiArray {
        guard let ctx = CGContext(
            data: nil, width: size, height: size,
            bitsPerComponent: 8, bytesPerRow: size * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            throw CMZError.inferenceFailed(reason: "letterbox context")
        }
        ctx.setFillColor(gray: 0.5, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))
        ctx.draw(image, in: CGRect(x: padX, y: padY, width: scaledW, height: scaledH))
        guard let pixels = ctx.data else {
            throw CMZError.inferenceFailed(reason: "letterbox pixels")
        }

        let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: size), NSNumber(value: size)],
                                   dataType: .float32)
        let dst = arr.dataPointer.assumingMemoryBound(to: Float.self)
        let src = pixels.assumingMemoryBound(to: UInt8.self)
        let hw = size * size
        let inv: Float = 1 / 255
        for i in 0..<hw {
            dst[0 * hw + i] = Float(src[i * 4 + 0]) * inv
            dst[1 * hw + i] = Float(src[i * 4 + 1]) * inv
            dst[2 * hw + i] = Float(src[i * 4 + 2]) * inv
        }
        return arr
    }

    private static func load(modelId: String, containing needle: String,
                              compute: MLComputeUnits) async throws -> MLModel {
        let dir = CMZPaths.modelDir(id: modelId)
        guard FileManager.default.fileExists(atPath: CMZPaths.metaFile(modelId: modelId).path) else {
            throw CMZError.modelNotInstalled(id: modelId)
        }
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)) ?? []
        guard let url = entries.first(where: {
            ($0.pathExtension == "mlpackage" || $0.pathExtension == "mlmodelc")
                && $0.lastPathComponent.lowercased().contains(needle)
        }) else {
            throw CMZError.inferenceFailed(reason: "sub-model '\(needle)' missing")
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = compute
        return try await MLModel.load(contentsOf: url, configuration: cfg)
    }

    private static func loadTokenizer(modelId: String) throws -> CLIPTokenizer {
        let dir = CMZPaths.modelDir(id: modelId)
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)) ?? []
        guard let vocab = entries.first(where: {
            $0.pathExtension == "json"
                && $0.lastPathComponent.lowercased().contains("clip_vocab")
        }) ?? entries.first(where: { $0.pathExtension == "json" }) else {
            throw CMZError.inferenceFailed(reason: "clip_vocab.json missing")
        }
        return try CLIPTokenizer(vocabularyURL: vocab)
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

    private static func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let iw = max(0, min(a.maxX, b.maxX) - max(a.minX, b.minX))
        let ih = max(0, min(a.maxY, b.maxY) - max(a.minY, b.minY))
        let inter = Float(iw * ih)
        let union = Float(a.width * a.height) + Float(b.width * b.height) - inter
        return union > 0 ? inter / union : 0
    }
}
