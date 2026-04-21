import Foundation
import CoreGraphics

/// Mask-based image inpainting.
///
/// Input: original RGB image + single-channel mask where white = erase/fill.
/// Output: the same image with masked regions synthesized.
///
/// - Note: v1-alpha scaffolding. Full implementation will port pre/post-processing
///   from `conversion_scripts/convert_lama.py` / `convert_aot_gan.py`.
public struct InpaintRequest: CMZRequest {

    public enum Model: String, Sendable, CaseIterable {
        case lama = "lama"
        case aotGan = "aot_gan_inpainting"
    }

    public struct Input: Sendable {
        public var image: CGImage
        public var mask: CGImage
        public init(image: CGImage, mask: CGImage) {
            self.image = image; self.mask = mask
        }
    }

    public let model: Model
    public var computeUnits: CMZComputeUnits

    public init(model: Model = .lama, computeUnits: CMZComputeUnits = .auto) {
        self.model = model
        self.computeUnits = computeUnits
    }

    public var modelId: String { model.rawValue }

    public func perform(on input: Input) async throws -> CGImage {
        throw CMZError.inferenceFailed(reason: "InpaintRequest is scaffolded in v1-alpha")
    }
}
