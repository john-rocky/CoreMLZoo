import Foundation
import CoreML

/// Shared helpers for locating + loading compiled mlpackages/mlmodelc from
/// `CMZModelStore`.
enum ModelLoading {

    /// Find a `.mlpackage` / `.mlmodelc` inside the model's install directory.
    ///
    /// The manifest doesn't prescribe filenames, so we look up by expected
    /// name first, then fall back to the first mlpackage we find. Most models
    /// have a single package; multi-package models (MatAnyone, Florence-2,
    /// Hyper-SD) must specify `name`.
    static func locate(modelId: String, name: String? = nil) throws -> URL {
        let dir = try CMZModelStore.shared.localDirectoryBlocking(for: modelId)
        let fm = FileManager.default
        if let name {
            let direct = dir.appendingPathComponent(name)
            if fm.fileExists(atPath: direct.path) { return direct }
            let packaged = dir.appendingPathComponent("\(name).mlpackage")
            if fm.fileExists(atPath: packaged.path) { return packaged }
            let compiled = dir.appendingPathComponent("\(name).mlmodelc")
            if fm.fileExists(atPath: compiled.path) { return compiled }
        }
        guard let entries = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else {
            throw CMZError.modelNotInstalled(id: modelId)
        }
        if let pkg = entries.first(where: { $0.pathExtension == "mlpackage" }) { return pkg }
        if let pkg = entries.first(where: { $0.pathExtension == "mlmodelc" }) { return pkg }
        throw CMZError.modelNotInstalled(id: modelId)
    }

    static func load(modelId: String,
                     subName: String? = nil,
                     compute: MLComputeUnits) async throws -> MLModel {
        let url = try locate(modelId: modelId, name: subName)
        let config = MLModelConfiguration()
        config.computeUnits = compute
        // Compile on-the-fly if we got an .mlpackage. MLModel(contentsOf:) on
        // iOS accepts an uncompiled .mlpackage and will compile+cache
        // automatically on first load, so no extra step is needed here.
        do {
            return try await MLModel.load(contentsOf: url, configuration: config)
        } catch {
            throw CMZError.inferenceFailed(reason: "load \(url.lastPathComponent): \(error)")
        }
    }
}

// Synchronous accessor used above; the store's public `localDirectory(for:)`
// is isolated to the actor so we need a bridge for non-async call sites.
extension CMZModelStore {
    nonisolated func localDirectoryBlocking(for id: String) throws -> URL {
        let path = CMZPaths.modelDir(id: id)
        let meta = CMZPaths.metaFile(modelId: id)
        guard FileManager.default.fileExists(atPath: meta.path) else {
            throw CMZError.modelNotInstalled(id: id)
        }
        return path
    }
}
