import Foundation
import CoreML
import CoreGraphics
import CoreImage
import Accelerate

/// Per-frame video matting. Default: MatAnyone (5 mlpackages, 111 MB FP16
/// total).
///
/// The SDK owns the ring-buffer state machine (`mem_key`, `mem_shrinkage`,
/// `mem_msk_value`, `mem_valid`, `sensory`, `obj_memory`, last frame
/// snapshots). Callers see only per-frame input/output.
///
/// Resolution is locked to 768×432 landscape — portrait sources should be
/// pre-rotated by the caller (the Hub App's video compositor rotates
/// automatically but a session-level API keeps that concern explicit).
///
/// ```swift
/// let session = try await VideoMattingSession(
///     firstFrame: landscapeFrame,
///     firstFrameMask: binarisedMask
/// )
/// for frame in nextFrames {
///     let alpha = try await session.process(frame)
///     // composite alpha over your chosen background
/// }
/// ```
public final class VideoMattingSession: CMZSession {

    public enum Model: Sendable {
        case matAnyone
        var id: String { "matanyone" }
    }

    public let model: Model
    public var modelIds: [String] { [model.id] }

    // Input resolution — must match the converted mlpackages.
    public static let frameWidth = 768
    public static let frameHeight = 432

    // Engine constants mirroring the converter.
    static let stride       = 16
    static let queryHeight  = frameHeight / stride   // 27
    static let queryWidth   = frameWidth  / stride   // 48
    static let queryHW      = queryHeight * queryWidth // 1296
    static let memMaxFrames = 5
    static let memCapacity  = memMaxFrames * queryHW   // 6480
    static let memEvery     = 5
    static let valueDim     = 256
    static let keyDim       = 64
    static let sensoryDim   = 256
    static let querySlots   = 16
    static let summaryDim   = 257

    // MARK: - Models

    private let encoder: MLModel
    private let maskEncoder: MLModel
    private let readFirst: MLModel
    private let read: MLModel
    private let decoder: MLModel

    // MARK: - Ring buffer state

    private let sensory: MLMultiArray
    private let lastMask: MLMultiArray
    private let lastPixFeat: MLMultiArray
    private let lastMskValue: MLMultiArray
    private let objMemory: MLMultiArray
    private let memKey: MLMultiArray
    private let memShrinkage: MLMultiArray
    private let memMskValue: MLMultiArray
    private let memValid: MLMultiArray

    private var currentFrame: Int = -1
    private var lastMemFrame: Int = 0
    private var nextFifoSlot: Int = 1

    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // MARK: - Init

    public init(model: Model = .matAnyone,
                firstFrame: CGImage,
                firstFrameMask: CGImage,
                computeUnits: CMZComputeUnits = .auto) async throws {
        self.model = model
        let modelId = model.id
        // Per MatAnyoneDemo: encoder/mask_encoder/decoder run on cpuAndGPU,
        // read/read_first must stay cpuOnly (iOS GPU crashes on the
        // singleton num_objects slice).
        async let enc      = Self.loadModule(modelId: modelId, substring: "encoder",
                                              exclude: "mask", units: .cpuAndGPU)
        async let maskEnc  = Self.loadModule(modelId: modelId, substring: "mask_encoder",
                                              exclude: nil, units: .cpuAndGPU)
        async let rdFirst  = Self.loadModule(modelId: modelId, substring: "read_first",
                                              exclude: nil, units: .cpuOnly)
        async let rd       = Self.loadModule(modelId: modelId, substring: "read",
                                              exclude: "first", units: .cpuOnly)
        async let dec      = Self.loadModule(modelId: modelId, substring: "decoder",
                                              exclude: nil, units: .cpuAndGPU)
        self.encoder     = try await enc
        self.maskEncoder = try await maskEnc
        self.readFirst   = try await rdFirst
        self.read        = try await rd
        self.decoder     = try await dec

        // State arrays.
        self.sensory = try MLMultiArray(shape: [1, 1,
            NSNumber(value: Self.sensoryDim),
            NSNumber(value: Self.queryHeight),
            NSNumber(value: Self.queryWidth)], dataType: .float32)
        self.lastMask = try MLMultiArray(shape: [1, 1,
            NSNumber(value: Self.frameHeight),
            NSNumber(value: Self.frameWidth)], dataType: .float32)
        self.lastPixFeat = try MLMultiArray(shape: [1,
            NSNumber(value: Self.valueDim),
            NSNumber(value: Self.queryHeight),
            NSNumber(value: Self.queryWidth)], dataType: .float32)
        self.lastMskValue = try MLMultiArray(shape: [1, 1,
            NSNumber(value: Self.valueDim),
            NSNumber(value: Self.queryHeight),
            NSNumber(value: Self.queryWidth)], dataType: .float32)
        self.objMemory = try MLMultiArray(shape: [1, 1, 1,
            NSNumber(value: Self.querySlots),
            NSNumber(value: Self.summaryDim)], dataType: .float32)
        self.memKey = try MLMultiArray(shape: [1,
            NSNumber(value: Self.keyDim),
            NSNumber(value: Self.memCapacity)], dataType: .float32)
        self.memShrinkage = try MLMultiArray(shape: [1, 1,
            NSNumber(value: Self.memCapacity)], dataType: .float32)
        self.memMskValue = try MLMultiArray(shape: [1,
            NSNumber(value: Self.valueDim),
            NSNumber(value: Self.memCapacity)], dataType: .float32)
        self.memValid = try MLMultiArray(shape: [1,
            NSNumber(value: Self.memCapacity)], dataType: .float32)
        Self.zero(self.sensory);     Self.zero(self.lastMask)
        Self.zero(self.lastPixFeat); Self.zero(self.lastMskValue)
        Self.zero(self.objMemory);   Self.zero(self.memKey)
        Self.zero(self.memShrinkage); Self.zero(self.memMskValue)
        Self.zero(self.memValid)

        // Seed with the first frame + mask.
        let firstImageArray = try buildImageArray(firstFrame)
        let firstMaskArray = try buildMaskArray(firstFrameMask)
        _ = try await step(image: firstImageArray,
                            providedMask: firstMaskArray,
                            firstFramePred: true)
    }

    // MARK: - Public API

    /// Process the next frame. Returns an 8-bit grayscale alpha mask at
    /// 768×432; composite it over the original-resolution frame yourself.
    public func process(_ frame: CGImage) async throws -> CGImage {
        let imageArray = try buildImageArray(frame)
        let alpha = try await step(image: imageArray, providedMask: nil, firstFramePred: false)
        return try Self.alphaToCGImage(alpha,
                                       width: Self.frameWidth,
                                       height: Self.frameHeight)
    }

    /// Reset the ring buffer. Use when starting a new clip without
    /// constructing a new Session.
    public func reset(firstFrame: CGImage, firstFrameMask: CGImage) async throws {
        Self.zero(sensory);     Self.zero(lastMask)
        Self.zero(lastPixFeat); Self.zero(lastMskValue)
        Self.zero(objMemory);   Self.zero(memKey)
        Self.zero(memShrinkage); Self.zero(memMskValue)
        Self.zero(memValid)
        currentFrame = -1; lastMemFrame = 0; nextFifoSlot = 1

        let firstImageArray = try buildImageArray(firstFrame)
        let firstMaskArray = try buildMaskArray(firstFrameMask)
        _ = try await step(image: firstImageArray,
                            providedMask: firstMaskArray,
                            firstFramePred: true)
    }

    // MARK: - Engine step (ported from MatAnyoneHubEngine)

    private func step(image: MLMultiArray,
                      providedMask: MLMultiArray? = nil,
                      firstFramePred: Bool = false) async throws -> [Float] {
        let isMemFrame: Bool
        let needSegment: Bool
        if firstFramePred {
            currentFrame = 0
            lastMemFrame = 0
            isMemFrame = true
            needSegment = true
        } else {
            currentFrame += 1
            isMemFrame = ((currentFrame - lastMemFrame) >= Self.memEvery) || (providedMask != nil)
            needSegment = providedMask == nil
        }

        let encOut = try await encoder.prediction(from: dict(["image": image]))
        let f16 = try feature(encOut, "f16")
        let f8  = try feature(encOut, "f8")
        let f4  = try feature(encOut, "f4")
        let f2  = try feature(encOut, "f2")
        let f1  = try feature(encOut, "f1")
        let pixFeat = try feature(encOut, "pix_feat")
        let key       = try feature(encOut, "key")
        let shrinkage = try feature(encOut, "shrinkage")
        let selection = try feature(encOut, "selection")

        var alphaArray = [Float](repeating: 0,
                                  count: Self.frameHeight * Self.frameWidth)
        if needSegment {
            let memReadout: MLMultiArray
            if currentFrame == 0 {
                let rfOut = try await readFirst.prediction(from: dict([
                    "pix_feat": pixFeat,
                    "last_msk_value": lastMskValue,
                    "sensory": sensory,
                    "last_mask": lastMask,
                    "obj_memory": objMemory,
                ]))
                memReadout = try feature(rfOut, "mem_readout")
            } else {
                let rdOut = try await read.prediction(from: dict([
                    "query_key": key,
                    "query_selection": selection,
                    "pix_feat": pixFeat,
                    "sensory": sensory,
                    "last_mask": lastMask,
                    "last_pix_feat": lastPixFeat,
                    "last_msk_value": lastMskValue,
                    "mem_key": memKey,
                    "mem_shrinkage": memShrinkage,
                    "mem_msk_value": memMskValue,
                    "mem_valid": memValid,
                    "obj_memory": objMemory,
                ]))
                memReadout = try feature(rdOut, "mem_readout")
            }

            let decOut = try await decoder.prediction(from: dict([
                "f16": f16, "f8": f8, "f4": f4, "f2": f2, "f1": f1,
                "mem_readout": memReadout,
                "sensory": sensory,
            ]))
            let newSensory = try feature(decOut, "new_sensory")
            let alpha      = try feature(decOut, "alpha")
            Self.copy(alpha, into: &alphaArray)
            Self.copyContents(from: newSensory, to: sensory)
            Self.copyContents(from: alpha, to: lastMask)
        } else if let m = providedMask {
            Self.copyContents(from: m, to: lastMask)
            Self.copy(m, into: &alphaArray)
        }

        Self.copyContents(from: pixFeat, to: lastPixFeat)

        let meOut = try await maskEncoder.prediction(from: dict([
            "image": image,
            "pix_feat": pixFeat,
            "sensory": sensory,
            "mask": lastMask,
        ]))
        let mskValue = try feature(meOut, "mask_value")
        let newSensoryME = try feature(meOut, "new_sensory")
        let objSummary = try feature(meOut, "obj_summary")
        Self.copyContents(from: mskValue, to: lastMskValue)

        if isMemFrame {
            if firstFramePred {
                Self.zero(memValid); Self.zero(objMemory); nextFifoSlot = 1
            }
            let slot: Int
            if providedMask != nil || firstFramePred {
                slot = 0
            } else {
                slot = nextFifoSlot
                nextFifoSlot += 1
                if nextFifoSlot >= Self.memMaxFrames { nextFifoSlot = 1 }
            }
            writeMemorySlot(slot: slot, key: key, shrinkage: shrinkage, mskValue: mskValue)
            accumulateObjSummary(objSummary)
            Self.copyContents(from: newSensoryME, to: sensory)
            lastMemFrame = currentFrame
        }
        return alphaArray
    }

    // MARK: - Memory bookkeeping

    private func writeMemorySlot(slot: Int,
                                  key: MLMultiArray,
                                  shrinkage: MLMultiArray,
                                  mskValue: MLMultiArray) {
        let HW = Self.queryHW
        let start = slot * HW
        let inK = Self.dense(key)
        let inS = Self.dense(shrinkage)
        let inM = Self.dense(mskValue)

        let kPtr = memKey.dataPointer.assumingMemoryBound(to: Float.self)
        for c in 0..<Self.keyDim {
            let dst = c * Self.memCapacity + start
            let src = c * HW
            for i in 0..<HW { kPtr[dst + i] = inK[src + i] }
        }
        let sPtr = memShrinkage.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<HW { sPtr[start + i] = inS[i] }
        let mPtr = memMskValue.dataPointer.assumingMemoryBound(to: Float.self)
        for c in 0..<Self.valueDim {
            let dst = c * Self.memCapacity + start
            let src = c * HW
            for i in 0..<HW { mPtr[dst + i] = inM[src + i] }
        }
        let vPtr = memValid.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<HW { vPtr[start + i] = 1.0 }
    }

    private func accumulateObjSummary(_ summary: MLMultiArray) {
        let src = Self.dense(summary)
        let dst = objMemory.dataPointer.assumingMemoryBound(to: Float.self)
        let n = Self.querySlots * Self.summaryDim
        var any: Float = 0
        for i in 0..<n { any += abs(dst[i]) }
        if any == 0 {
            for i in 0..<n { dst[i] = src[i] }
        } else {
            for i in 0..<n { dst[i] += src[i] }
        }
    }

    // MARK: - Input / output conversion

    private func buildImageArray(_ image: CGImage) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, 3,
            NSNumber(value: Self.frameHeight),
            NSNumber(value: Self.frameWidth)], dataType: .float32)
        let buffer = try ImageBuffer.bgraBuffer(from: image,
            size: CGSize(width: Self.frameWidth, height: Self.frameHeight))
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw CMZError.invalidInput(reason: "pixel buffer base")
        }
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        let src = base.assumingMemoryBound(to: UInt8.self)
        let dst = arr.dataPointer.assumingMemoryBound(to: Float.self)
        let inv: Float = 1 / 255
        let w = Self.frameWidth, h = Self.frameHeight
        let plane = w * h
        for y in 0..<h {
            for x in 0..<w {
                let p = y * rowBytes + x * 4
                // BGRA → R, G, B planes
                dst[0 * plane + y * w + x] = Float(src[p + 2]) * inv
                dst[1 * plane + y * w + x] = Float(src[p + 1]) * inv
                dst[2 * plane + y * w + x] = Float(src[p + 0]) * inv
            }
        }
        return arr
    }

    private func buildMaskArray(_ mask: CGImage) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, 1,
            NSNumber(value: Self.frameHeight),
            NSNumber(value: Self.frameWidth)], dataType: .float32)
        let w = Self.frameWidth, h = Self.frameHeight
        var bytes = [UInt8](repeating: 0, count: w * h)
        let ctx = CGContext(data: &bytes, width: w, height: h,
                            bitsPerComponent: 8, bytesPerRow: w,
                            space: CGColorSpaceCreateDeviceGray(),
                            bitmapInfo: CGImageAlphaInfo.none.rawValue)
        ctx?.interpolationQuality = .none  // binary mask — avoid smoothing
        ctx?.draw(mask, in: CGRect(x: 0, y: 0, width: w, height: h))
        let dst = arr.dataPointer.assumingMemoryBound(to: Float.self)
        let inv: Float = 1 / 255
        for i in 0..<(w * h) {
            // Binarise at 0.5 — MatAnyone seed masks are expected binary
            let v = Float(bytes[i]) * inv
            dst[i] = v > 0.5 ? 1 : 0
        }
        return arr
    }

    private static func alphaToCGImage(_ alpha: [Float], width: Int, height: Int) throws -> CGImage {
        var bytes = [UInt8](repeating: 0, count: width * height)
        for i in 0..<(width * height) {
            bytes[i] = UInt8(clamping: Int(max(0, min(1, alpha[i])) * 255))
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let cg = CGImage(
                width: width, height: height,
                bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: width,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                provider: provider, decode: nil,
                shouldInterpolate: true, intent: .defaultIntent) else {
            throw CMZError.inferenceFailed(reason: "alpha → CGImage")
        }
        return cg
    }

    // MARK: - Loading

    private static func loadModule(modelId: String,
                                    substring: String,
                                    exclude: String?,
                                    units: MLComputeUnits) async throws -> MLModel {
        let dir = CMZPaths.modelDir(id: modelId)
        guard FileManager.default.fileExists(atPath: CMZPaths.metaFile(modelId: modelId).path) else {
            throw CMZError.modelNotInstalled(id: modelId)
        }
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)) ?? []
        let lowerSub = substring.lowercased()
        let candidates = entries.filter {
            guard $0.pathExtension == "mlpackage" || $0.pathExtension == "mlmodelc" else { return false }
            let lower = $0.lastPathComponent.lowercased()
            if !lower.contains(lowerSub) { return false }
            if let excl = exclude, lower.contains(excl.lowercased()) { return false }
            return true
        }
        guard let url = candidates.first else {
            throw CMZError.inferenceFailed(reason: "MatAnyone sub-model '\(substring)' missing")
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units
        return try await MLModel.load(contentsOf: url, configuration: cfg)
    }

    // MARK: - Low-level utilities

    private func dict(_ pairs: [String: MLMultiArray]) -> MLDictionaryFeatureProvider {
        var d = [String: MLFeatureValue]()
        for (k, v) in pairs { d[k] = MLFeatureValue(multiArray: v) }
        return try! MLDictionaryFeatureProvider(dictionary: d)
    }

    private func feature(_ provider: MLFeatureProvider, _ name: String) throws -> MLMultiArray {
        guard let v = provider.featureValue(for: name)?.multiArrayValue else {
            throw CMZError.inferenceFailed(reason: "MatAnyone missing feature '\(name)'")
        }
        return v
    }

    static func zero(_ array: MLMultiArray) {
        let p = array.dataPointer.assumingMemoryBound(to: Float.self)
        memset(p, 0, array.count * MemoryLayout<Float>.size)
    }

    /// Stride-safe read of an MLMultiArray into a contiguous Float buffer.
    static func readFloats(_ src: MLMultiArray,
                            into dst: UnsafeMutablePointer<Float>,
                            count: Int) {
        let shape = src.shape.map { $0.intValue }
        let strides = src.strides.map { $0.intValue }
        let srcPtr = src.dataPointer.assumingMemoryBound(to: Float.self)

        var expected = 1, dense = true
        for i in (0..<shape.count).reversed() {
            if strides[i] != expected { dense = false; break }
            expected *= shape[i]
        }
        if dense {
            memcpy(dst, srcPtr, count * MemoryLayout<Float>.size); return
        }
        let n = shape.count
        var counters = [Int](repeating: 0, count: n)
        for i in 0..<count {
            var offset = 0
            for d in 0..<n { offset += counters[d] * strides[d] }
            dst[i] = srcPtr[offset]
            var d = n - 1
            while d >= 0 {
                counters[d] += 1
                if counters[d] < shape[d] { break }
                counters[d] = 0; d -= 1
            }
        }
    }

    static func dense(_ src: MLMultiArray) -> [Float] {
        let count = src.count
        let buf = UnsafeMutablePointer<Float>.allocate(capacity: count)
        defer { buf.deallocate() }
        readFloats(src, into: buf, count: count)
        return Array(UnsafeBufferPointer(start: buf, count: count))
    }

    static func copyContents(from src: MLMultiArray, to dst: MLMultiArray) {
        precondition(src.count == dst.count, "shape mismatch")
        let dPtr = dst.dataPointer.assumingMemoryBound(to: Float.self)
        readFloats(src, into: dPtr, count: src.count)
    }

    static func copy(_ src: MLMultiArray, into dst: inout [Float]) {
        let n = min(src.count, dst.count)
        dst.withUnsafeMutableBufferPointer { buf in
            readFloats(src, into: buf.baseAddress!, count: n)
        }
    }
}
