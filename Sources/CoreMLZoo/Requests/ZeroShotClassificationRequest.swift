import Foundation
import CoreGraphics

/// Open-vocabulary image classification via image/text embedding models
/// (SigLIP). Caller provides candidate captions at inference time.
///
/// - Note: v1-alpha scaffolding. Ports from `sample_apps/SigLIPDemo`.
///   SigLIP requires (0.5, 0.5) normalization — not ImageNet.
public struct ZeroShotClassificationRequest: CMZRequest {

    public enum Model: String, Sendable, CaseIterable {
        case siglipBase = "siglip_image_encoder"
    }

    public struct ClassifiedLabel: Sendable {
        public let label: String
        public let score: Float
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

    public init(model: Model = .siglipBase, computeUnits: CMZComputeUnits = .auto) {
        self.model = model; self.computeUnits = computeUnits
    }

    /// SigLIP ships as 2 mlpackages (image encoder + text encoder) + tokenizer.
    /// `modelId` is the image encoder; the paired text encoder/tokenizer live
    /// in the same manifest entry's `files[]`.
    public var modelId: String { model.rawValue }

    public func perform(on input: Input) async throws -> [ClassifiedLabel] {
        throw CMZError.inferenceFailed(reason: "ZeroShotClassificationRequest is scaffolded in v1-alpha")
    }
}
