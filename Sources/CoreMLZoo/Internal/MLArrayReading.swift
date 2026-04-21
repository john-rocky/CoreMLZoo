import Foundation
import CoreML
import Accelerate

/// Stride-aware readers for MLMultiArray.
///
/// On `.all` / `.cpuAndNeuralEngine`, the ANE pads rows for SIMD alignment,
/// so `dataPointer` is NOT C-contiguous. These helpers honor `strides[]`
/// and use `vImageConvert_Planar16FtoPlanarF` for FP16→FP32 row-at-a-time
/// so the cost is ~free.
enum MLArrayReading {

    /// Read a (1, H, W) array as row-major `[Float]` of length H*W.
    static func read2D(_ array: MLMultiArray,
                       height: Int,
                       width: Int) -> [Float] {
        let strides = array.strides.map { $0.intValue }
        let rowStride = strides[1]
        var result = [Float](repeating: 0, count: height * width)

        if array.dataType == .float16 {
            let src = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            result.withUnsafeMutableBufferPointer { dst in
                for r in 0..<height {
                    var srcBuf = vImage_Buffer(
                        data: UnsafeMutableRawPointer(mutating: src + r * rowStride),
                        height: 1, width: vImagePixelCount(width),
                        rowBytes: width * 2)
                    var dstBuf = vImage_Buffer(
                        data: dst.baseAddress! + r * width,
                        height: 1, width: vImagePixelCount(width),
                        rowBytes: width * 4)
                    vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
                }
            }
        } else {
            let src = array.dataPointer.assumingMemoryBound(to: Float.self)
            result.withUnsafeMutableBufferPointer { dst in
                for r in 0..<height {
                    memcpy(dst.baseAddress! + r * width,
                           src + r * rowStride,
                           width * 4)
                }
            }
        }
        return result
    }

    /// Read a (1, H, W, C) array as interleaved `[Float]` of length H*W*C.
    static func read3D(_ array: MLMultiArray,
                       height: Int,
                       width: Int,
                       channels: Int) -> [Float] {
        let strides = array.strides.map { $0.intValue }
        let rowStride = strides[1]
        let colStride = strides[2]
        let chStride = strides[3]
        let rowElements = width * channels
        var result = [Float](repeating: 0, count: height * width * channels)
        let interleaved = (colStride == channels && chStride == 1)

        if array.dataType == .float16 {
            let src = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            if interleaved {
                result.withUnsafeMutableBufferPointer { dst in
                    for r in 0..<height {
                        var srcBuf = vImage_Buffer(
                            data: UnsafeMutableRawPointer(mutating: src + r * rowStride),
                            height: 1, width: vImagePixelCount(rowElements),
                            rowBytes: rowElements * 2)
                        var dstBuf = vImage_Buffer(
                            data: dst.baseAddress! + r * rowElements,
                            height: 1, width: vImagePixelCount(rowElements),
                            rowBytes: rowElements * 4)
                        vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
                    }
                }
            } else {
                for r in 0..<height {
                    let baseR = r * rowStride
                    for c in 0..<width {
                        let baseC = baseR + c * colStride
                        let dst = (r * width + c) * channels
                        for ch in 0..<channels {
                            result[dst + ch] = Float(Float16(bitPattern: src[baseC + ch * chStride]))
                        }
                    }
                }
            }
        } else {
            let src = array.dataPointer.assumingMemoryBound(to: Float.self)
            if interleaved {
                result.withUnsafeMutableBufferPointer { dst in
                    for r in 0..<height {
                        memcpy(dst.baseAddress! + r * rowElements,
                               src + r * rowStride,
                               rowElements * 4)
                    }
                }
            } else {
                for r in 0..<height {
                    let baseR = r * rowStride
                    for c in 0..<width {
                        let baseC = baseR + c * colStride
                        let dst = (r * width + c) * channels
                        for ch in 0..<channels {
                            result[dst + ch] = src[baseC + ch * chStride]
                        }
                    }
                }
            }
        }
        return result
    }
}
