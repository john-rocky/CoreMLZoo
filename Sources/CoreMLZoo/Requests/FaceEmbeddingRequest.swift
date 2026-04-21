import Foundation
import CoreML
import CoreGraphics
import Accelerate

/// 512-dim face identity embedding. Use `cosineSimilarity` to compare two
/// embeddings; ≥ ~0.4 is the same identity for AdaFace at default settings.
///
/// Unlike `Vision.VNFaceObservation` which returns landmarks/quality only,
/// this is an actual identity embedding suitable for verification /
/// clustering. Default model: AdaFace IR-18 (48 MB).
///
/// `perform` expects a pre-aligned 112×112 face crop. Do detection +
/// 5-point alignment upstream (Vision's `VNDetectFaceLandmarksRequest` is
/// fine for the detection half).
public struct FaceEmbeddingRequest: CMZRequest {

    public enum Model: String, Sendable, CaseIterable {
        case adaFaceIR18 = "adaface_ir18"

        var inputSize: Int { 112 }
    }

    public let model: Model
    public var computeUnits: CMZComputeUnits

    public init(model: Model = .adaFaceIR18, computeUnits: CMZComputeUnits = .auto) {
        self.model = model; self.computeUnits = computeUnits
    }

    public var modelId: String { model.rawValue }

    public func perform(on input: CGImage) async throws -> [Float] {
        let size = model.inputSize
        let buffer = try ImageBuffer.bgraBuffer(from: input,
                                                 size: CGSize(width: size, height: size))

        let compute = (computeUnits == .auto) ? MLComputeUnits.all
                                              : computeUnits.mlComputeUnits
        let coreModel = try await ModelLoading.load(modelId: modelId, compute: compute)

        let inputKey = coreModel.modelDescription.inputDescriptionsByName.keys.first ?? "image"
        let provider = try MLDictionaryFeatureProvider(
            dictionary: [inputKey: MLFeatureValue(pixelBuffer: buffer)])
        let output: MLFeatureProvider
        do {
            output = try await coreModel.prediction(from: provider)
        } catch {
            throw CMZError.inferenceFailed(reason: "\(error)")
        }

        for (name, desc) in coreModel.modelDescription.outputDescriptionsByName
            where desc.type == .multiArray {
            if let arr = output.featureValue(for: name)?.multiArrayValue {
                return extractFloats(arr)
            }
        }
        throw CMZError.inferenceFailed(reason: "no embedding output")
    }

    private func extractFloats(_ array: MLMultiArray) -> [Float] {
        let n = array.count
        var result = [Float](repeating: 0, count: n)
        if array.dataType == .float16 {
            let src = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            result.withUnsafeMutableBufferPointer { dst in
                var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src),
                                      height: 1, width: vImagePixelCount(n),
                                      rowBytes: n * 2)
                var d = vImage_Buffer(data: dst.baseAddress!,
                                      height: 1, width: vImagePixelCount(n),
                                      rowBytes: n * 4)
                vImageConvert_Planar16FtoPlanarF(&s, &d, 0)
            }
        } else {
            memcpy(&result, array.dataPointer, n * 4)
        }
        return result
    }

    public static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "embedding dim mismatch")
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in a.indices {
            dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
        }
        return dot / max(1e-6, (na.squareRoot() * nb.squareRoot()))
    }
}
