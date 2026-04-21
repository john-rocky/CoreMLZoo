import Foundation
import CoreML
import CoreImage
import CoreGraphics

public struct BackgroundRemovalResult: Sendable {
    /// Input-resolution cutout with transparent background.
    public let cutout: CGImage
    /// Grayscale alpha mask at the original image's resolution.
    public let mask: CGImage
}

/// Model-agnostic background removal. Default model is RMBG-1.4 (42 MB INT8).
///
/// Usage:
/// ```swift
/// let req = BackgroundRemovalRequest()
/// let out = try await req.prepareAndPerform(on: inputCGImage)
/// ```
public struct BackgroundRemovalRequest: CMZRequest {
    public enum Model: String, Sendable, CaseIterable {
        case rmbg14 = "rmbg_1_4"

        /// Input spatial size expected by the model.
        var inputSize: Int {
            switch self { case .rmbg14: return 1024 }
        }

        /// Whether the output alpha mask needs min-max renormalization to
        /// [0, 1]. RMBG-1.4's raw output is unbounded sigmoid logits; the
        /// upstream repo applies min-max stretch post-sigmoid.
        var requiresMinMaxStretch: Bool {
            switch self { case .rmbg14: return true }
        }

        var computeUnits: CMZComputeUnits {
            switch self { case .rmbg14: return .cpuOnly }
        }
    }

    public let model: Model
    public var computeUnits: CMZComputeUnits

    public init(model: Model = .rmbg14, computeUnits: CMZComputeUnits = .auto) {
        self.model = model
        self.computeUnits = computeUnits
    }

    public var modelId: String { model.rawValue }

    public func perform(on input: CGImage) async throws -> BackgroundRemovalResult {
        let size = model.inputSize
        let buffer = try ImageBuffer.bgraBuffer(from: input,
                                                size: CGSize(width: size, height: size))

        let units = (computeUnits == .auto) ? model.computeUnits.mlComputeUnits
                                            : computeUnits.mlComputeUnits
        let coreModel = try await ModelLoading.load(modelId: modelId, compute: units)

        let provider = try MLDictionaryFeatureProvider(dictionary: ["image": buffer])
        let output: MLFeatureProvider
        do {
            output = try await coreModel.prediction(from: provider)
        } catch {
            throw CMZError.inferenceFailed(reason: "\(error)")
        }
        guard let array = output.featureValue(for: "alpha_mask")?.multiArrayValue else {
            throw CMZError.inferenceFailed(reason: "alpha_mask feature missing")
        }
        let maskFloats = Self.extractMaskFloats(array,
                                                stretch: model.requiresMinMaxStretch)

        return try Self.makeResult(maskFloats: maskFloats,
                                   modelSize: size,
                                   source: input)
    }

    // MARK: - Internals

    private static func extractMaskFloats(_ array: MLMultiArray,
                                          stretch: Bool) -> [Float] {
        let count = array.count
        var raw: [Float]
        if array.dataType == .float16 {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            raw = (0..<count).map { Float(ptr[$0]) }
        } else {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float32.self)
            raw = (0..<count).map { ptr[$0] }
        }
        if stretch {
            let mi = raw.min() ?? 0
            let ma = raw.max() ?? 1
            let range = ma - mi
            if range > 1e-6 {
                for i in raw.indices { raw[i] = (raw[i] - mi) / range }
            }
        }
        return raw
    }

    private static func makeResult(maskFloats: [Float],
                                   modelSize: Int,
                                   source: CGImage) throws -> BackgroundRemovalResult {
        // Pack mask floats → 8-bit gray CGImage
        var bytes = [UInt8](repeating: 0, count: modelSize * modelSize)
        for i in bytes.indices {
            bytes[i] = UInt8(clamping: Int(maskFloats[i] * 255))
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let maskSmall = CGImage(width: modelSize, height: modelSize,
                                      bitsPerComponent: 8, bitsPerPixel: 8,
                                      bytesPerRow: modelSize,
                                      space: CGColorSpaceCreateDeviceGray(),
                                      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                      provider: provider, decode: nil,
                                      shouldInterpolate: true, intent: .defaultIntent) else {
            throw CMZError.inferenceFailed(reason: "mask image assembly")
        }

        let origCI = CIImage(cgImage: source)
        let extent = origCI.extent
        let sx = extent.width / CGFloat(modelSize)
        let sy = extent.height / CGFloat(modelSize)
        let maskCI = CIImage(cgImage: maskSmall)
            .transformed(by: CGAffineTransform(scaleX: sx, y: sy))

        let transparent = CIImage.empty().cropped(to: extent)
        let blended = origCI.applyingFilter("CIBlendWithMask", parameters: [
            kCIInputBackgroundImageKey: transparent,
            kCIInputMaskImageKey: maskCI,
        ])
        let ctx = CIContext(options: [.useSoftwareRenderer: false])
        guard let cutoutCG = ctx.createCGImage(blended, from: extent),
              let maskFullCG = ctx.createCGImage(maskCI, from: extent) else {
            throw CMZError.inferenceFailed(reason: "CIContext rendering")
        }
        return BackgroundRemovalResult(cutout: cutoutCG, mask: maskFullCG)
    }
}
