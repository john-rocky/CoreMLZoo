import Foundation
import CoreML
import CoreGraphics

/// 4× super-resolution. All variants take an RGB image and return an RGB
/// image at 4× the input dimensions.
///
/// - Note: Implementation is stubbed in v1-alpha; pre/post-processing will be
///   ported from `sample_apps/SinSRDemo` (SinSR uses 3-model pipeline) and
///   the Real-ESRGAN conversion notes.
public struct UpscaleRequest: CMZRequest {

    public enum Model: String, Sendable, CaseIterable {
        case realESRGAN4x = "real_esrgan_4x"
        case realESRGANAnime4x = "real_esrgan_anime_4x"
        case esrgan = "esrgan"
        case ultraSharp = "ultrasharp"
        case sinSR = "sinsr_denoiser"  // pipeline: encoder → denoiser → decoder
        case bsrgan = "bsrgan"
        case aesrgan = "a_esrgan"

        var scaleFactor: Int { 4 }
    }

    public let model: Model
    public var computeUnits: CMZComputeUnits

    public init(model: Model = .realESRGAN4x, computeUnits: CMZComputeUnits = .auto) {
        self.model = model
        self.computeUnits = computeUnits
    }

    public var modelId: String { model.rawValue }

    public func perform(on input: CGImage) async throws -> CGImage {
        throw CMZError.inferenceFailed(reason: "UpscaleRequest is scaffolded in v1-alpha")
    }
}
