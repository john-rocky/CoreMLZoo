import Foundation
import CoreGraphics

/// 512-dim face identity embedding. Use cosine distance to compare two
/// embeddings (same identity ≥ ~0.4).
///
/// Unlike `Vision.VNFaceObservation` which returns only landmarks/quality,
/// this is an actual identity embedding suitable for verification / clustering.
/// Default model: AdaFace IR-18 (48 MB).
///
/// - Note: v1-alpha scaffolding. Ports from `sample_apps/AdaFaceDemo`.
public struct FaceEmbeddingRequest: CMZRequest {

    public enum Model: String, Sendable, CaseIterable {
        case adaFaceIR18 = "adaface_ir18"

        /// Expected input size for the face crop (112×112 RGB, aligned).
        var inputSize: Int { 112 }
    }

    public let model: Model
    public var computeUnits: CMZComputeUnits

    public init(model: Model = .adaFaceIR18, computeUnits: CMZComputeUnits = .auto) {
        self.model = model; self.computeUnits = computeUnits
    }

    public var modelId: String { model.rawValue }

    /// `input` is a pre-aligned face crop (caller is responsible for face
    /// detection + 5-point alignment, typically via Vision's
    /// `VNDetectFaceLandmarksRequest`).
    public func perform(on input: CGImage) async throws -> [Float] {
        throw CMZError.inferenceFailed(reason: "FaceEmbeddingRequest is scaffolded in v1-alpha")
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
