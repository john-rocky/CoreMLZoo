import Foundation
import CoreGraphics

// MARK: - Low-light enhancement

/// Low-light image enhancement. Takes an RGB image, returns a brightened
/// version with denoising applied.
///
/// - Note: v1-alpha scaffolding.
public struct LowLightEnhanceRequest: CMZRequest {
    public enum Model: String, Sendable, CaseIterable {
        case retinexformerFiveK = "retinexformer_fivek"
        case retinexformerNTIRE = "retinexformer_ntire"
        case stableLLVE = "stablellve"
        case zeroDCE = "zero_dce"
    }

    public let model: Model
    public var computeUnits: CMZComputeUnits

    public init(model: Model = .retinexformerFiveK, computeUnits: CMZComputeUnits = .auto) {
        self.model = model; self.computeUnits = computeUnits
    }

    public var modelId: String { model.rawValue }

    public func perform(on input: CGImage) async throws -> CGImage {
        throw CMZError.inferenceFailed(reason: "LowLightEnhanceRequest is scaffolded in v1-alpha")
    }
}

// MARK: - Image restoration

/// Image restoration (deblur / denoise / derain / contrast / general).
///
/// - Note: v1-alpha scaffolding.
public struct ImageRestorationRequest: CMZRequest {
    public enum Model: String, Sendable, CaseIterable {
        case mprnetDeblurring = "mprnet_deblurring"
        case mprnetDenoising = "mprnet_denoising"
        case mprnetDeraining = "mprnet_deraining"
        case mirnetv2Denoising = "mirnetv2_denoising"
        case mirnetv2ContrastEnhancement = "mirnetv2_contrast_enhancement"
        case mirnetv2SuperResolution = "mirnetv2_super_resolution"
        case mirnetv2LowLight = "mirnetv2_low_light_enhancement"
    }

    public let model: Model
    public var computeUnits: CMZComputeUnits

    public init(model: Model, computeUnits: CMZComputeUnits = .auto) {
        self.model = model; self.computeUnits = computeUnits
    }

    public var modelId: String { model.rawValue }

    public func perform(on input: CGImage) async throws -> CGImage {
        throw CMZError.inferenceFailed(reason: "ImageRestorationRequest is scaffolded in v1-alpha")
    }
}

// MARK: - Colorization

/// Grayscale → color image. Default: DDColor Tiny (Lab→ab residual).
///
/// - Note: v1-alpha scaffolding. Ports from `sample_apps/DDColorDemo`.
public struct ColorizeRequest: CMZRequest {
    public enum Model: String, Sendable, CaseIterable {
        case ddColorTiny = "ddcolor_tiny"
    }

    public let model: Model
    public var computeUnits: CMZComputeUnits

    public init(model: Model = .ddColorTiny, computeUnits: CMZComputeUnits = .auto) {
        self.model = model; self.computeUnits = computeUnits
    }

    public var modelId: String { model.rawValue }

    public func perform(on input: CGImage) async throws -> CGImage {
        throw CMZError.inferenceFailed(reason: "ColorizeRequest is scaffolded in v1-alpha")
    }
}
