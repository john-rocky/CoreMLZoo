import Foundation
import CoreGraphics

public struct DetectedObject: Sendable {
    public let label: String
    public let confidence: Float
    public let boundingBox: CGRect  // normalized [0,1]^4, Vision convention (origin top-left)
}

/// Open-vocabulary object detection — caller supplies the list of class
/// labels at inference time. Current model: YOLO-World V2-S (detector +
/// CLIP ViT-B/32 text encoder + BPE vocab, 3 files).
///
/// - Note: v1-alpha scaffolding. Ports from `sample_apps/YOLOWorldDemo`.
public struct OpenVocabDetectionRequest: CMZRequest {

    public enum Model: String, Sendable, CaseIterable {
        case yoloWorldV2S = "yoloworld_detector"
    }

    public struct Input: Sendable {
        public var image: CGImage
        public var classes: [String]
        public var confidenceThreshold: Float
        public var iouThreshold: Float
        public init(image: CGImage, classes: [String],
                    confidenceThreshold: Float = 0.25,
                    iouThreshold: Float = 0.45) {
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

    /// YOLO-World bundles detector + text encoder + BPE vocab into a single
    /// manifest entry. `modelId` points at the detector; the other two files
    /// are downloaded alongside it.
    public var modelId: String { model.rawValue }

    public func perform(on input: Input) async throws -> [DetectedObject] {
        throw CMZError.inferenceFailed(reason: "OpenVocabDetectionRequest is scaffolded in v1-alpha")
    }
}
