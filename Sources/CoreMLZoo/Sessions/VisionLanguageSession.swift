import Foundation
import CoreGraphics

/// Vision-language multitask session. Default: Florence-2 (vision encoder +
/// text encoder + decoder, ~260 MB INT8). Handles captioning, OCR,
/// detection-by-prompt, and grounding in one model.
///
/// Usage:
/// ```swift
/// let session = try await VisionLanguageSession(model: .florence2)
/// let caption = try await session.caption(cg)
/// let regions = try await session.groundPhrase("the red car", in: cg)
/// let text = try await session.ocr(cg)
/// ```
///
/// - Note: v1-alpha scaffolding. SDK owns the seq2seq decoding loop
///   (no KV cache — re-runs full sequence each step, matching the
///   converted mlpackages). Ports from `sample_apps/Florence2Demo`.
public final class VisionLanguageSession: CMZSession {

    public enum Model: Sendable {
        case florence2Base
        var ids: [String] {
            switch self {
            case .florence2Base:
                return ["florence2_vision_encoder",
                        "florence2_text_encoder",
                        "florence2_decoder"]
            }
        }
    }

    public let model: Model
    public var modelIds: [String] { model.ids }

    public init(model: Model = .florence2Base,
                computeUnits: CMZComputeUnits = .auto) async throws {
        self.model = model
        _ = computeUnits
    }

    /// Short caption ("<CAPTION>" task token in Florence-2 terminology).
    public func caption(_ image: CGImage, detail: CaptionDetail = .brief) async throws -> String {
        throw CMZError.inferenceFailed(reason: "VisionLanguageSession is scaffolded in v1-alpha")
    }

    public enum CaptionDetail: Sendable { case brief, detailed, moreDetailed }

    /// OCR (all text in the image).
    public func ocr(_ image: CGImage) async throws -> String {
        throw CMZError.inferenceFailed(reason: "VisionLanguageSession is scaffolded in v1-alpha")
    }

    /// Phrase grounding. Returns bounding boxes for the regions matching `phrase`.
    public func groundPhrase(_ phrase: String, in image: CGImage) async throws -> [CGRect] {
        throw CMZError.inferenceFailed(reason: "VisionLanguageSession is scaffolded in v1-alpha")
    }
}
