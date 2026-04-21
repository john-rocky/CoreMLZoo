import Foundation
import CoreML
import CoreGraphics
import Accelerate

/// Mask-based image inpainting.
///
/// Input: original RGB image + single-channel mask where **white = erase/fill**,
/// black = keep. Output: same image with masked regions synthesized.
///
/// LaMa is fully convolutional — works at arbitrary sizes. The converted
/// mlpackage fixes it to 512×512; callers get back a 512×512 result and
/// should composite it over the original at their preferred resolution.
public struct InpaintRequest: CMZRequest {

    public enum Model: String, Sendable, CaseIterable {
        case lama = "lama"
        case aotGan = "aot_gan_inpainting"

        var inputSize: CGSize {
            switch self {
            case .lama:    return CGSize(width: 512, height: 512)
            case .aotGan:  return CGSize(width: 512, height: 512)
            }
        }
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
        self.model = model; self.computeUnits = computeUnits
    }

    public var modelId: String { model.rawValue }

    public func perform(on input: Input) async throws -> CGImage {
        let size = model.inputSize
        let resizedImage = ImagePixels.resized(input.image, to: size)
        let resizedMask = ImagePixels.resized(input.mask, to: size)
        let imageBuf = try ImageBuffer.bgraBuffer(from: resizedImage, size: size)
        let maskBuf = try ImageBuffer.bgraBuffer(from: resizedMask, size: size)

        let compute = (computeUnits == .auto) ? MLComputeUnits.all
                                              : computeUnits.mlComputeUnits
        let coreModel = try await ModelLoading.load(modelId: modelId, compute: compute)

        let inputNames = coreModel.modelDescription.inputDescriptionsByName
        // Try common naming conventions
        let imageKey = inputNames.keys.first(where: { $0.lowercased().contains("image") || $0 == "input" }) ?? "image"
        let maskKey = inputNames.keys.first(where: { $0.lowercased().contains("mask") }) ?? "mask"

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            imageKey: MLFeatureValue(pixelBuffer: imageBuf),
            maskKey:  MLFeatureValue(pixelBuffer: maskBuf),
        ])
        let output: MLFeatureProvider
        do {
            output = try await coreModel.prediction(from: provider)
        } catch {
            throw CMZError.inferenceFailed(reason: "\(error)")
        }

        for (name, desc) in coreModel.modelDescription.outputDescriptionsByName {
            if desc.type == .image,
               let pb = output.featureValue(for: name)?.imageBufferValue,
               let cg = ImageBuffer.cgImage(fromBGRA: pb) {
                return cg
            }
            if desc.type == .multiArray,
               let arr = output.featureValue(for: name)?.multiArrayValue {
                return try chwFloatsToCGImage(arr)
            }
        }
        throw CMZError.inferenceFailed(reason: "no usable output feature")
    }

    private func chwFloatsToCGImage(_ array: MLMultiArray) throws -> CGImage {
        let shape = array.shape.map { $0.intValue }
        guard shape.count == 4, shape[1] == 3 else {
            throw CMZError.inferenceFailed(reason: "expected (1,3,H,W), got \(shape)")
        }
        let h = shape[2], w = shape[3], plane = h * w
        var floats = [Float](repeating: 0, count: 3 * plane)
        if array.dataType == .float16 {
            let src = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            floats.withUnsafeMutableBufferPointer { dst in
                var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src),
                                      height: 1, width: vImagePixelCount(3 * plane),
                                      rowBytes: 3 * plane * 2)
                var d = vImage_Buffer(data: dst.baseAddress!,
                                      height: 1, width: vImagePixelCount(3 * plane),
                                      rowBytes: 3 * plane * 4)
                vImageConvert_Planar16FtoPlanarF(&s, &d, 0)
            }
        } else {
            memcpy(&floats, array.dataPointer, 3 * plane * 4)
        }
        var rgba = [UInt8](repeating: 255, count: plane * 4)
        for i in 0..<plane {
            rgba[i * 4]     = UInt8(clamping: Int(max(0, min(1, floats[0 * plane + i])) * 255))
            rgba[i * 4 + 1] = UInt8(clamping: Int(max(0, min(1, floats[1 * plane + i])) * 255))
            rgba[i * 4 + 2] = UInt8(clamping: Int(max(0, min(1, floats[2 * plane + i])) * 255))
        }
        guard let cg = ImagePixels.cgImage(fromRGBA: rgba, width: w, height: h) else {
            throw CMZError.inferenceFailed(reason: "CGImage assembly")
        }
        return cg
    }
}
