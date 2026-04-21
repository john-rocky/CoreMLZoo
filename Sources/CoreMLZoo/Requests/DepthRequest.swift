import Foundation
import CoreML
import CoreGraphics

public struct DepthResult: Sendable {
    /// Depth in meters (when the model is metric) or relative disparity.
    /// Row-major, `size × size`.
    public let depth: [Float]
    /// Optional per-pixel surface normals (x, y, z in [-1, 1]) interleaved.
    /// `size × size × 3` or empty if the model doesn't produce normals.
    public let normal: [Float]
    /// Optional confidence mask in [0, 1]. Empty if not produced.
    public let mask: [Float]
    public let size: Int
    public let depthMin: Float
    public let depthMax: Float
    public let isMetric: Bool
}

/// Monocular depth estimation. Default is MoGe-2 (ViT-B, 504×504, metric).
///
/// ```swift
/// let req = DepthRequest(model: .moGe2)
/// let out = try await req.prepareAndPerform(on: cg)
/// ```
public struct DepthRequest: CMZRequest {

    public enum Model: String, Sendable, CaseIterable {
        case moGe2 = "moge2_vitb_normal_504"
        case depthAnythingV3Small = "depth_anything_v3_small"
        case midasSmall = "midas_small"

        var inputSize: Int {
            switch self {
            case .moGe2, .depthAnythingV3Small: return 504
            case .midasSmall: return 256
            }
        }

        var computeUnits: CMZComputeUnits {
            // ViT at 504 fits comfortably on ANE; MiDaS small is MobileNet-ish.
            .all
        }

        var hasNormal: Bool { self == .moGe2 }
        var hasMask: Bool { self == .moGe2 }
        var isMetric: Bool { self == .moGe2 }
    }

    public let model: Model
    public var computeUnits: CMZComputeUnits

    public init(model: Model = .moGe2, computeUnits: CMZComputeUnits = .auto) {
        self.model = model
        self.computeUnits = computeUnits
    }

    public var modelId: String { model.rawValue }

    public func perform(on input: CGImage) async throws -> DepthResult {
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
        return try Self.decode(output, model: model, size: size)
    }

    private static func decode(_ output: MLFeatureProvider,
                               model: Model,
                               size: Int) throws -> DepthResult {
        // Per-model feature names. When we add new depth models, extend here.
        let depthKey: String
        switch model {
        case .moGe2:                 depthKey = "depth"
        case .depthAnythingV3Small:  depthKey = "depth"
        case .midasSmall:            depthKey = "relative_depth"
        }
        guard let depthArr = output.featureValue(for: depthKey)?.multiArrayValue else {
            throw CMZError.inferenceFailed(reason: "'\(depthKey)' feature missing")
        }
        var depth = MLArrayReading.read2D(depthArr, height: size, width: size)

        var normal: [Float] = []
        var mask: [Float] = []
        var isMetric = model.isMetric

        if model.hasMask,
           let mArr = output.featureValue(for: "mask")?.multiArrayValue {
            mask = MLArrayReading.read2D(mArr, height: size, width: size)
        }
        if model.hasNormal,
           let nArr = output.featureValue(for: "normal")?.multiArrayValue {
            normal = MLArrayReading.read3D(nArr, height: size, width: size, channels: 3)
        }
        if model.isMetric,
           let scaleArr = output.featureValue(for: "metric_scale")?.multiArrayValue {
            let scale = scaleArr[0].floatValue
            for i in depth.indices { depth[i] *= scale }
        } else {
            isMetric = false
        }

        var dMin: Float = .greatestFiniteMagnitude
        var dMax: Float = 0
        if mask.isEmpty {
            for v in depth { if v < dMin { dMin = v }; if v > dMax { dMax = v } }
        } else {
            for i in depth.indices where mask[i] > 0.5 {
                let v = depth[i]
                if v < dMin { dMin = v }
                if v > dMax { dMax = v }
            }
            if dMin == .greatestFiniteMagnitude { dMin = 0; dMax = 1 }
        }

        return DepthResult(depth: depth, normal: normal, mask: mask,
                           size: size, depthMin: dMin, depthMax: dMax,
                           isMetric: isMetric)
    }
}
