import Foundation
import CoreML
import CoreGraphics

/// 4× super-resolution. Takes any input size that the converted mlpackage
/// accepts and returns a 4×-larger CGImage.
///
/// Note: The Real-ESRGAN family of mlpackages was converted with **fixed**
/// input shapes (see `conversion_scripts/convert_realesrgan.py`) so callers
/// must pre-resize/crop to the model's expected tile size. The SDK uses
/// whatever size the compiled model advertises.
///
/// SinSR is a 3-model pipeline (encoder → denoiser → decoder) — currently
/// only the denoiser is wired; encoder/decoder scaffolding will land in a
/// follow-up.
public struct UpscaleRequest: CMZRequest {

    public enum Model: String, Sendable, CaseIterable {
        case realESRGAN = "realesrgan"
        /// Face-aware super-resolution (v1 face portraits primarily).
        case gfpgan = "gfpgan"

        var scaleFactor: Int { 4 }
    }

    public let model: Model
    public var computeUnits: CMZComputeUnits

    public init(model: Model = .realESRGAN, computeUnits: CMZComputeUnits = .auto) {
        self.model = model
        self.computeUnits = computeUnits
    }

    public var modelId: String { model.rawValue }

    public func perform(on input: CGImage) async throws -> CGImage {
        let inputSize = try await Self.inputSize(modelId: modelId)
        return try await SimpleImage2Image.run(modelId: modelId,
                                                input: input,
                                                inputSize: inputSize,
                                                computeUnits: computeUnits)
    }

    /// Read the compiled mlpackage's ImageType constraint to discover the
    /// fixed input shape. Falls back to 256×256 when the description can't
    /// be introspected.
    private static func inputSize(modelId: String) async throws -> CGSize {
        let url = try ModelLoading.locate(modelId: modelId)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        let model = try await MLModel.load(contentsOf: url, configuration: config)
        if let desc = model.modelDescription.inputDescriptionsByName.values.first,
           desc.type == .image,
           let imageConstraint = desc.imageConstraint {
            return CGSize(width: imageConstraint.pixelsWide,
                          height: imageConstraint.pixelsHigh)
        }
        return CGSize(width: 256, height: 256)
    }
}
