import Foundation
import CoreML
import CoreGraphics

/// Grayscale → color image. Default: DDColor Tiny (Lab→ab residual, 512×512
/// input). Output resolution matches input.
///
/// The SDK runs LAB conversion on a concurrent per-row queue so that full-res
/// photos (12MP+) colorize in a fraction of a second after inference.
public struct ColorizeRequest: CMZRequest {

    public enum Model: String, Sendable, CaseIterable {
        case ddColor = "ddcolor"

        var inputSize: Int { 512 }
    }

    public let model: Model
    public var computeUnits: CMZComputeUnits

    public init(model: Model = .ddColor, computeUnits: CMZComputeUnits = .auto) {
        self.model = model; self.computeUnits = computeUnits
    }

    public var modelId: String { model.rawValue }

    public func perform(on input: CGImage) async throws -> CGImage {
        let origW = input.width, origH = input.height
        let size = model.inputSize

        // 1. Extract L channel at full resolution (kept as-is)
        let (rgb, _, _) = ImagePixels.rgbFloats(from: input)
        var origL = [Float](repeating: 0, count: origW * origH)
        DispatchQueue.concurrentPerform(iterations: origH) { y in
            let row = y * origW
            for x in 0..<origW {
                let i = row + x
                let (l, _, _) = LabColor.srgbToLab(r: rgb[i * 3],
                                                    g: rgb[i * 3 + 1],
                                                    b: rgb[i * 3 + 2])
                origL[i] = l
            }
        }

        // 2. Resize to 512x512 and turn it into gray-RGB (L, 0, 0 → sRGB)
        let resized = ImagePixels.resized(input, to: CGSize(width: size, height: size))
        let (smallRGB, _, _) = ImagePixels.rgbFloats(from: resized)
        let plane = size * size
        let grayCHW = try MLMultiArray(shape: [1, 3, size, size] as [NSNumber],
                                       dataType: .float32)
        let fp = grayCHW.dataPointer.assumingMemoryBound(to: Float.self)
        DispatchQueue.concurrentPerform(iterations: plane) { i in
            let (l, _, _) = LabColor.srgbToLab(r: smallRGB[i * 3],
                                                g: smallRGB[i * 3 + 1],
                                                b: smallRGB[i * 3 + 2])
            let (gr, gg, gb) = LabColor.labToSrgb(l: l, a: 0, b: 0)
            fp[0 * plane + i] = gr
            fp[1 * plane + i] = gg
            fp[2 * plane + i] = gb
        }

        // 3. Run inference
        let compute = (computeUnits == .auto) ? MLComputeUnits.all : computeUnits.mlComputeUnits
        let coreModel = try await ModelLoading.load(modelId: modelId, compute: compute)
        let provider = try MLDictionaryFeatureProvider(
            dictionary: ["image": MLFeatureValue(multiArray: grayCHW)])
        let output: MLFeatureProvider
        do {
            output = try await coreModel.prediction(from: provider)
        } catch {
            throw CMZError.inferenceFailed(reason: "\(error)")
        }
        guard let ab = output.featureValue(for: "ab_channels")?.multiArrayValue else {
            throw CMZError.inferenceFailed(reason: "ab_channels missing")
        }

        // 4. Pull AB (2×H×W) into planar floats
        var ab512 = [Float](repeating: 0, count: ab.count)
        let abPtr = ab.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<ab.count { ab512[i] = abPtr[i] }

        // 5. Upscale AB to original resolution
        let abFull = ImagePixels.resize2ChannelPlanar(ab512,
                                                      fromW: size, fromH: size,
                                                      toW: origW, toH: origH)

        // 6. Merge L + AB → sRGB
        let pixelCount = origW * origH
        var rgba = [UInt8](repeating: 255, count: pixelCount * 4)
        DispatchQueue.concurrentPerform(iterations: pixelCount) { i in
            let (r, g, b) = LabColor.labToSrgb(l: origL[i],
                                                a: abFull[i],
                                                b: abFull[pixelCount + i])
            rgba[i * 4]     = UInt8(clamping: Int(r * 255))
            rgba[i * 4 + 1] = UInt8(clamping: Int(g * 255))
            rgba[i * 4 + 2] = UInt8(clamping: Int(b * 255))
        }
        guard let out = ImagePixels.cgImage(fromRGBA: rgba, width: origW, height: origH) else {
            throw CMZError.inferenceFailed(reason: "failed to assemble output image")
        }
        return out
    }
}
