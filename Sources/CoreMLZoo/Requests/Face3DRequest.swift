import Foundation
import CoreGraphics

/// Monocular 3D face reconstruction (3DDFA V2). Returns 3DMM parameters +
/// per-vertex 3D positions.
///
/// - Note: v1-alpha scaffolding. Ports from `sample_apps/Face3DDemo`.
public struct Face3DRequest: CMZRequest {

    public enum Model: String, Sendable, CaseIterable {
        case ddfaV2 = "face3d_3ddfa_v2"
    }

    public struct Result: Sendable {
        public let vertices: [Float]          // (N, 3) flattened
        public let rotation: [Float]          // 3x3
        public let translation: [Float]       // 3
        public let scale: Float
    }

    public let model: Model
    public var computeUnits: CMZComputeUnits

    public init(model: Model = .ddfaV2, computeUnits: CMZComputeUnits = .auto) {
        self.model = model; self.computeUnits = computeUnits
    }

    public var modelId: String { model.rawValue }

    /// `input` is a pre-cropped face region (caller uses Vision for detection).
    public func perform(on input: CGImage) async throws -> Result {
        throw CMZError.inferenceFailed(reason: "Face3DRequest is scaffolded in v1-alpha")
    }
}
