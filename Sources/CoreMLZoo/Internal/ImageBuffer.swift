import Foundation
import CoreVideo
import CoreGraphics
import CoreImage
import Accelerate

/// Helpers for converting between `CGImage` / `CIImage` and `CVPixelBuffer`
/// shapes expected by Core ML (BGRA / grayscale, specific sizes).
enum ImageBuffer {

    /// Resize a `CGImage` and render into a BGRA CVPixelBuffer suitable for
    /// `MLFeatureValue(pixelBuffer:)`.
    static func bgraBuffer(from cg: CGImage,
                           size: CGSize) throws -> CVPixelBuffer {
        let width = Int(size.width), height = Int(size.height)
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                          kCVPixelFormatType_32BGRA,
                                          attrs as CFDictionary, &pb)
        guard status == kCVReturnSuccess, let buffer = pb else {
            throw CMZError.invalidInput(reason: "CVPixelBufferCreate failed (\(status))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                  width: width, height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                              | CGBitmapInfo.byteOrder32Little.rawValue) else {
            throw CMZError.invalidInput(reason: "CGContext creation failed")
        }
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }

    /// Render a single-channel grayscale CVPixelBuffer (0..1 fp32) to a CGImage.
    static func cgImage(from buffer: CVPixelBuffer) -> CGImage? {
        let ci = CIImage(cvPixelBuffer: buffer)
        let ctx = CIContext()
        return ctx.createCGImage(ci, from: ci.extent)
    }

    /// Render a BGRA CVPixelBuffer to a CGImage.
    static func cgImage(fromBGRA buffer: CVPixelBuffer) -> CGImage? {
        cgImage(from: buffer)
    }
}
