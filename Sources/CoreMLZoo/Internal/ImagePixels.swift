import Foundation
import CoreGraphics
import Accelerate
import CoreML

/// Raw RGB / grayscale pixel access used by color-space-heavy requests
/// (DDColor Lab conversion, etc.).
enum ImagePixels {

    /// Return RGB floats in [0, 1], row-major, interleaved (rgbrgb…).
    static func rgbFloats(from image: CGImage) -> (pixels: [Float], width: Int, height: Int) {
        let w = image.width, h = image.height
        var raw = [UInt8](repeating: 0, count: w * h * 4)
        let ctx = CGContext(
            data: &raw, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        var out = [Float](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            out[i * 3]     = Float(raw[i * 4])     / 255
            out[i * 3 + 1] = Float(raw[i * 4 + 1]) / 255
            out[i * 3 + 2] = Float(raw[i * 4 + 2]) / 255
        }
        return (out, w, h)
    }

    /// Resize a `CGImage` with high-quality interpolation.
    static func resized(_ image: CGImage, to size: CGSize) -> CGImage {
        let w = Int(size.width), h = Int(size.height)
        let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage()!
    }

    /// Build a CGImage from interleaved RGBA bytes.
    static func cgImage(fromRGBA pixels: [UInt8], width: Int, height: Int) -> CGImage? {
        var mutable = pixels
        let ctx = CGContext(
            data: &mutable, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        return ctx?.makeImage()
    }

    /// Planar `[2, H, W]` float bilinear resize using vImage per channel.
    static func resize2ChannelPlanar(_ src: [Float],
                                      fromW: Int, fromH: Int,
                                      toW: Int, toH: Int) -> [Float] {
        let srcCount = fromW * fromH
        let dstCount = toW * toH
        var result = [Float](repeating: 0, count: 2 * dstCount)
        result.withUnsafeMutableBufferPointer { dstBuf in
            src.withUnsafeBufferPointer { srcBuf in
                let srcBase = UnsafeMutablePointer(mutating: srcBuf.baseAddress!)
                let dstBase = dstBuf.baseAddress!
                for ch in 0..<2 {
                    var srcImg = vImage_Buffer(
                        data: srcBase.advanced(by: ch * srcCount),
                        height: vImagePixelCount(fromH),
                        width: vImagePixelCount(fromW),
                        rowBytes: fromW * MemoryLayout<Float>.stride)
                    var dstImg = vImage_Buffer(
                        data: dstBase.advanced(by: ch * dstCount),
                        height: vImagePixelCount(toH),
                        width: vImagePixelCount(toW),
                        rowBytes: toW * MemoryLayout<Float>.stride)
                    vImageScale_PlanarF(&srcImg, &dstImg, nil,
                                        vImage_Flags(kvImageHighQualityResampling))
                }
            }
        }
        return result
    }

    /// Fill an MLMultiArray shaped (1, 3, H, W) with a CHW planar RGB from
    /// interleaved RGB floats in [0, 1].
    static func fillCHW(_ array: MLMultiArray,
                        fromInterleaved rgb: [Float],
                        channels: Int = 3,
                        width: Int,
                        height: Int) {
        let plane = width * height
        let fp = array.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<plane {
            for c in 0..<channels {
                fp[c * plane + i] = rgb[i * channels + c]
            }
        }
    }
}
