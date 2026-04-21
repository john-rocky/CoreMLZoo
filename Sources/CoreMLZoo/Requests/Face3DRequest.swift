import Foundation
import CoreML
import CoreGraphics
import Accelerate

/// 3D face pose + expression parameters from a face crop. Default: 3DDFA V2
/// (6.3 MB, 120×120 input). Returns the 3×3 rotation matrix, Euler angles
/// (yaw/pitch/roll in radians), and 10-dim expression coefficients.
///
/// `perform` expects a face crop sized per 3DDFA V2's convention: square ROI
/// expanded 1.58× around the face rect with the center shifted 14% upward
/// (toward the forehead), cropped to 120×120. The `cropROI(faceRect:image:)`
/// helper performs this for you.
public struct Face3DRequest: CMZRequest {

    public enum Model: String, Sendable, CaseIterable {
        case ddfaV2 = "face3d_3ddfa_v2"

        var inputSize: Int { 120 }
    }

    public struct Result: Sendable {
        public let rotation: [[Float]]       // 3x3
        public let yaw: Float
        public let pitch: Float
        public let roll: Float
        public let expressionCoeffs: [Float] // length 10
    }

    public let model: Model
    public var computeUnits: CMZComputeUnits

    public init(model: Model = .ddfaV2, computeUnits: CMZComputeUnits = .auto) {
        self.model = model; self.computeUnits = computeUnits
    }

    public var modelId: String { model.rawValue }

    public func perform(on input: CGImage) async throws -> Result {
        let size = model.inputSize
        let buffer = try ImageBuffer.bgraBuffer(from: input,
                                                 size: CGSize(width: size, height: size))
        let compute = (computeUnits == .auto) ? MLComputeUnits.all
                                              : computeUnits.mlComputeUnits
        let coreModel = try await ModelLoading.load(modelId: modelId, compute: compute)
        let inputKey = coreModel.modelDescription.inputDescriptionsByName.keys.first ?? "face_image"
        let provider = try MLDictionaryFeatureProvider(
            dictionary: [inputKey: MLFeatureValue(pixelBuffer: buffer)])
        let output: MLFeatureProvider
        do {
            output = try await coreModel.prediction(from: provider)
        } catch {
            throw CMZError.inferenceFailed(reason: "\(error)")
        }

        // Find the output array (62 params — pose + shape + expression)
        var params: [Float] = []
        for (name, desc) in coreModel.modelDescription.outputDescriptionsByName
            where desc.type == .multiArray {
            if let arr = output.featureValue(for: name)?.multiArrayValue, arr.count == 62 {
                params = extractFloats(arr); break
            }
        }
        guard params.count == 62 else {
            throw CMZError.inferenceFailed(reason: "expected 62-dim output, got \(params.count)")
        }

        // Pose is the first 12 params as a flattened 3x4 [R | t] matrix.
        let pose = Array(params[0..<12])
        var R: [[Float]] = [
            [pose[0], pose[1], pose[2]],
            [pose[4], pose[5], pose[6]],
            [pose[8], pose[9], pose[10]],
        ]
        for i in 0..<3 {
            let n = (R[i][0] * R[i][0] + R[i][1] * R[i][1] + R[i][2] * R[i][2]).squareRoot()
            if n > 1e-6 { R[i][0] /= n; R[i][1] /= n; R[i][2] /= n }
        }
        let (yaw, pitch, roll) = Self.euler(R)
        let expression = Array(params[52..<62])
        return Result(rotation: R, yaw: yaw, pitch: pitch, roll: roll,
                      expressionCoeffs: expression)
    }

    /// 3DDFA V2's crop convention. Caller supplies a face bounding box in
    /// Vision coordinates (origin bottom-left, normalized) OR image
    /// coordinates (origin top-left, pixels) — set `isNormalized` accordingly.
    public static func cropROI(faceRect: CGRect,
                                from image: CGImage,
                                isNormalized: Bool = true) -> CGImage? {
        let w = CGFloat(image.width), h = CGFloat(image.height)
        let left: CGFloat, right: CGFloat, top: CGFloat, bottom: CGFloat
        if isNormalized {
            left = faceRect.origin.x * w
            right = (faceRect.origin.x + faceRect.width) * w
            top = (1.0 - faceRect.origin.y - faceRect.height) * h
            bottom = (1.0 - faceRect.origin.y) * h
        } else {
            left = faceRect.origin.x
            right = faceRect.origin.x + faceRect.width
            top = faceRect.origin.y
            bottom = faceRect.origin.y + faceRect.height
        }
        let old = ((right - left) + (bottom - top)) / 2
        let cx = (left + right) / 2
        let cy = (top + bottom) / 2 - old * 0.14
        let size = old * 1.58
        let roi = CGRect(x: cx - size / 2, y: cy - size / 2, width: size, height: size)
            .intersection(CGRect(x: 0, y: 0, width: w, height: h))
        guard roi.width > 0, roi.height > 0 else { return nil }
        return image.cropping(to: roi)
    }

    private func extractFloats(_ array: MLMultiArray) -> [Float] {
        let n = array.count
        var out = [Float](repeating: 0, count: n)
        if array.dataType == .float16 {
            let src = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            out.withUnsafeMutableBufferPointer { dst in
                var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src),
                                      height: 1, width: vImagePixelCount(n), rowBytes: n * 2)
                var d = vImage_Buffer(data: dst.baseAddress!,
                                      height: 1, width: vImagePixelCount(n), rowBytes: n * 4)
                vImageConvert_Planar16FtoPlanarF(&s, &d, 0)
            }
        } else {
            memcpy(&out, array.dataPointer, n * 4)
        }
        return out
    }

    private static func euler(_ R: [[Float]]) -> (Float, Float, Float) {
        let sy = (R[0][0] * R[0][0] + R[1][0] * R[1][0]).squareRoot()
        if sy >= 1e-6 {
            return (atan2(-R[2][0], sy),             // yaw
                    atan2(R[2][1], R[2][2]),          // pitch
                    atan2(R[1][0], R[0][0]))          // roll
        }
        return (atan2(-R[2][0], sy),
                atan2(-R[1][2], R[1][1]),
                0)
    }
}
