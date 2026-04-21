import Foundation
import CoreML
import CoreGraphics
import Accelerate

/// Generic "image in, image out" inference runner used by multiple Requests
/// whose only per-model difference is the target input size and the output
/// feature name.
///
/// Assumptions:
/// - Input is named `image` (BGRA CVPixelBuffer) OR `input` (FP32 CHW)
/// - Output is named `image`, `output`, or `result` (FP32 CHW in [0, 1])
///
/// For models whose ImageType input accepts BGRA pixel buffers directly
/// (the most common case), we prefer pixel-buffer input because the
/// converter bakes `scale = 1/255` for us.
enum SimpleImage2Image {

    static func run(modelId: String,
                    input: CGImage,
                    inputSize: CGSize,
                    computeUnits: CMZComputeUnits) async throws -> CGImage {
        let compute = (computeUnits == .auto) ? MLComputeUnits.all
                                              : computeUnits.mlComputeUnits
        let coreModel = try await ModelLoading.load(modelId: modelId, compute: compute)

        let inputDesc = coreModel.modelDescription.inputDescriptionsByName
        let inputName = inputDesc.keys.first ?? "image"

        let provider: MLDictionaryFeatureProvider
        if let desc = inputDesc[inputName], desc.type == .image {
            let buffer = try ImageBuffer.bgraBuffer(from: input, size: inputSize)
            provider = try MLDictionaryFeatureProvider(
                dictionary: [inputName: MLFeatureValue(pixelBuffer: buffer)])
        } else {
            // Assume CHW float32, [1, 3, H, W] in [0, 1]
            let resized = ImagePixels.resized(input, to: inputSize)
            let (rgb, w, h) = ImagePixels.rgbFloats(from: resized)
            let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: h), NSNumber(value: w)],
                                        dataType: .float32)
            ImagePixels.fillCHW(arr, fromInterleaved: rgb, channels: 3, width: w, height: h)
            provider = try MLDictionaryFeatureProvider(
                dictionary: [inputName: MLFeatureValue(multiArray: arr)])
        }

        let output: MLFeatureProvider
        do {
            output = try await coreModel.prediction(from: provider)
        } catch {
            throw CMZError.inferenceFailed(reason: "\(error)")
        }

        // Pick the first feature that's an image or a (1,3,H,W) array.
        let outputNames = coreModel.modelDescription.outputDescriptionsByName
        for (name, desc) in outputNames {
            if desc.type == .image,
               let pb = output.featureValue(for: name)?.imageBufferValue {
                if let cg = ImageBuffer.cgImage(fromBGRA: pb) { return cg }
            }
            if desc.type == .multiArray,
               let arr = output.featureValue(for: name)?.multiArrayValue {
                return try imageFromCHW(arr)
            }
        }
        throw CMZError.inferenceFailed(reason: "no usable output feature")
    }

    /// Convert an FP16/FP32 (1, 3, H, W) array in [0, 1] into a CGImage.
    private static func imageFromCHW(_ array: MLMultiArray) throws -> CGImage {
        let shape = array.shape.map { $0.intValue }
        guard shape.count == 4, shape[1] == 3 else {
            throw CMZError.inferenceFailed(reason: "expected (1,3,H,W), got \(shape)")
        }
        let h = shape[2], w = shape[3]
        let plane = h * w

        var floats = [Float](repeating: 0, count: 3 * plane)
        if array.dataType == .float16 {
            let src = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            floats.withUnsafeMutableBufferPointer { dst in
                var srcBuf = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: src),
                    height: 1, width: vImagePixelCount(3 * plane),
                    rowBytes: 3 * plane * 2)
                var dstBuf = vImage_Buffer(
                    data: dst.baseAddress!,
                    height: 1, width: vImagePixelCount(3 * plane),
                    rowBytes: 3 * plane * 4)
                vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
            }
        } else {
            let src = array.dataPointer.assumingMemoryBound(to: Float.self)
            memcpy(&floats, src, 3 * plane * 4)
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
