import Foundation
import CoreML

/// Shared helpers for locating + loading compiled mlpackages/mlmodelc from
/// `CMZModelStore`.
///
/// **iOS Core ML mlpackage loading quirk:**
/// `MLModel.load(contentsOf: mlpackage, configuration:)` is advertised
/// as auto-compiling but in practice fails on some packages with
/// "Failed to open file ...coremldata.bin. It is not a valid .mlmodelc
/// file." The reliable path is to `MLModel.compileModel(at:)`
/// explicitly, move the resulting `.mlmodelc` next to the source, and
/// `MLModel(contentsOf:)` from there. We cache the compiled form on
/// disk so subsequent launches skip the compile step (which is the
/// expensive one — often 2-10 s per mlpackage).
///
/// **`.mlmodelc` preference:** when both `.mlmodelc` (compiled cache) and
/// `.mlpackage` (source) sit in the same directory, every lookup here
/// prefers the compiled form. Otherwise `FileManager.contentsOfDirectory`
/// order is unspecified, a `.mlpackage` can win the race, and every
/// subsequent load detours back through `compileModel` — emitting the
/// same "Failed to open coremldata.bin" probe noise on each launch even
/// though a cached `.mlmodelc` is sitting right next to it.
enum ModelLoading {

    /// Find a `.mlpackage` / `.mlmodelc` inside the model's install directory.
    ///
    /// The manifest doesn't prescribe filenames, so we look up by expected
    /// name first, then fall back to the first model file we find. Most models
    /// have a single package; multi-package models (MatAnyone, Florence-2,
    /// Hyper-SD) must specify `name`.
    static func locate(modelId: String, name: String? = nil) throws -> URL {
        let dir = try CMZModelStore.shared.localDirectoryBlocking(for: modelId)
        let fm = FileManager.default
        if let name {
            let direct = dir.appendingPathComponent(name)
            if fm.fileExists(atPath: direct.path) { return direct }
            let compiled = dir.appendingPathComponent("\(name).mlmodelc")
            if fm.fileExists(atPath: compiled.path) { return compiled }
            let packaged = dir.appendingPathComponent("\(name).mlpackage")
            if fm.fileExists(atPath: packaged.path) { return packaged }
        }
        guard let entries = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else {
            throw CMZError.modelNotInstalled(id: modelId)
        }
        if let pkg = entries.first(where: { $0.pathExtension == "mlmodelc" }) { return pkg }
        if let pkg = entries.first(where: { $0.pathExtension == "mlpackage" }) { return pkg }
        throw CMZError.modelNotInstalled(id: modelId)
    }

    /// Return all model files in the install directory whose filename
    /// contains `substring` (optionally excluding another substring),
    /// sorted so `.mlmodelc` entries come before `.mlpackage` entries.
    /// Used by multi-package models that pick sub-modules by name match.
    static func locateAll(modelId: String,
                           containing substring: String,
                           excluding: String? = nil,
                           caseInsensitive: Bool = true) throws -> [URL] {
        let dir = CMZPaths.modelDir(id: modelId)
        let fm = FileManager.default
        guard fm.fileExists(atPath: CMZPaths.metaFile(modelId: modelId).path) else {
            throw CMZError.modelNotInstalled(id: modelId)
        }
        let entries = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        let needle = caseInsensitive ? substring.lowercased() : substring
        let excl = excluding.map { caseInsensitive ? $0.lowercased() : $0 }
        let matches = entries.filter { url in
            guard url.pathExtension == "mlpackage" || url.pathExtension == "mlmodelc" else { return false }
            let name = caseInsensitive ? url.lastPathComponent.lowercased() : url.lastPathComponent
            if !name.contains(needle) { return false }
            if let excl, name.contains(excl) { return false }
            return true
        }
        return matches.sorted { a, b in
            // `.mlmodelc` wins over `.mlpackage`. Within an extension, keep
            // FileManager's natural order (stable-enough for single matches).
            if a.pathExtension == b.pathExtension { return false }
            return a.pathExtension == "mlmodelc"
        }
    }

    /// Convenience: locate the first matching sub-module and load it via
    /// `loadCompiled`. Throws `inferenceFailed` with `"\(label) '\(substring)' missing"`
    /// if nothing matches.
    static func loadSubmodel(modelId: String,
                              containing substring: String,
                              excluding: String? = nil,
                              caseInsensitive: Bool = true,
                              compute: MLComputeUnits,
                              missingLabel label: String) async throws -> MLModel {
        let candidates = try locateAll(modelId: modelId,
                                        containing: substring,
                                        excluding: excluding,
                                        caseInsensitive: caseInsensitive)
        guard let url = candidates.first else {
            throw CMZError.inferenceFailed(reason: "\(label) '\(substring)' missing")
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = compute
        return try await loadCompiled(at: url, configuration: cfg)
    }

    static func load(modelId: String,
                     subName: String? = nil,
                     compute: MLComputeUnits) async throws -> MLModel {
        let url = try locate(modelId: modelId, name: subName)
        let config = MLModelConfiguration()
        config.computeUnits = compute
        do {
            return try await loadCompiled(at: url, configuration: config)
        } catch {
            throw CMZError.inferenceFailed(reason: "load \(url.lastPathComponent): \(error)")
        }
    }

    /// Compile-and-cache aware loader. Accepts either `.mlpackage` or
    /// `.mlmodelc`. For `.mlpackage`, the compiled `.mlmodelc` is cached
    /// next to the source (same directory, same base name) so that
    /// second and subsequent launches skip the compile step entirely.
    static func loadCompiled(at url: URL,
                              configuration: MLModelConfiguration) async throws -> MLModel {
        if url.pathExtension == "mlmodelc" {
            return try MLModel(contentsOf: url, configuration: configuration)
        }

        let cachedCompiled = url
            .deletingPathExtension()
            .appendingPathExtension("mlmodelc")

        let fm = FileManager.default
        if fm.fileExists(atPath: cachedCompiled.path) {
            do {
                return try MLModel(contentsOf: cachedCompiled, configuration: configuration)
            } catch {
                // Cached compile is corrupt — remove and recompile below.
                try? fm.removeItem(at: cachedCompiled)
            }
        }

        // Compile. `MLModel.compileModel(at:)` writes to a temp location;
        // move it next to the source so the next launch hits the cache.
        let tempCompiled = try await MLModel.compileModel(at: url)
        do {
            if fm.fileExists(atPath: cachedCompiled.path) {
                try? fm.removeItem(at: cachedCompiled)
            }
            try fm.moveItem(at: tempCompiled, to: cachedCompiled)
            return try MLModel(contentsOf: cachedCompiled, configuration: configuration)
        } catch {
            // Move failed — load from the temp location anyway so this
            // prediction works; next launch will recompile.
            return try MLModel(contentsOf: tempCompiled, configuration: configuration)
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
