import Foundation
import CoreGraphics

/// Per-frame video matting. Default: MatAnyone (5 mlpackages, 111 MB FP16).
///
/// Usage:
/// ```swift
/// let session = try await VideoMattingSession(model: .matAnyone,
///                                              firstFrame: frame,
///                                              firstFrameMask: mask)
/// for try await frame in videoFrames {
///     let alpha = try await session.process(frame)
/// }
/// ```
///
/// - Note: v1-alpha scaffolding. The SDK will own the ring buffer
///   (`mem_key`, `mem_shrinkage`, `mem_msk_value`, `mem_valid`, `sensory`,
///   `obj_memory`) internally — callers never see mlpackage boundaries.
///   Ports from `sample_apps/MatAnyoneDemo`.
public final class VideoMattingSession: CMZSession {

    public enum Model: Sendable {
        case matAnyone
        var ids: [String] {
            switch self {
            case .matAnyone:
                return ["matanyone_encoder", "matanyone_mask_encoder",
                        "matanyone_read_first", "matanyone_read",
                        "matanyone_decoder"]
            }
        }
    }

    public let model: Model
    public var modelIds: [String] { model.ids }

    public init(model: Model = .matAnyone,
                firstFrame: CGImage,
                firstFrameMask: CGImage,
                computeUnits: CMZComputeUnits = .auto) async throws {
        self.model = model
        _ = firstFrame; _ = firstFrameMask; _ = computeUnits
        // TODO: port from MatAnyoneDemo.
    }

    /// Process the next frame, return an alpha mask at the same resolution.
    public func process(_ frame: CGImage) async throws -> CGImage {
        throw CMZError.inferenceFailed(reason: "VideoMattingSession is scaffolded in v1-alpha")
    }
}
