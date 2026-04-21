import Foundation

/// Central coordinator for the model manifest + on-disk cache.
///
/// Typical usage:
/// ```swift
/// // Ensure the model is installed before constructing a Request
/// for try await progress in CMZModelStore.shared.download(.depthAnythingV3Small) {
///     print("\(progress.phase) \(Int(progress.fraction * 100))%")
/// }
/// let result = try await DepthRequest(model: .depthAnythingV3Small).perform(on: image)
/// ```
///
/// The store is an `actor` because both the manifest cache and the
/// per-modelId downloader map are shared mutable state.
public actor CMZModelStore {

    public static let shared = CMZModelStore()

    /// Override the manifest URL (e.g. for staging / enterprise mirrors).
    public var manifestURL: URL = CMZPaths.defaultManifestURL

    private var cachedManifest: CMZManifest?
    private var downloaders: [String: FileDownloader] = [:]

    private init() {}

    // MARK: - Manifest

    /// Returns the manifest, preferring a fresh network fetch but falling
    /// back to the on-disk cache (or throwing if neither is available).
    public func manifest(forceRefresh: Bool = false) async throws -> CMZManifest {
        if !forceRefresh, let cached = cachedManifest { return cached }

        // Try disk cache first for instant return on offline launches.
        if !forceRefresh, let disk = try? readDiskCache() {
            cachedManifest = disk
            Task { try? await self.refreshFromNetwork() }
            return disk
        }
        return try await refreshFromNetwork()
    }

    @discardableResult
    private func refreshFromNetwork() async throws -> CMZManifest {
        do {
            let (data, _) = try await URLSession.shared.data(from: manifestURL)
            let decoded = try JSONDecoder().decode(CMZManifest.self, from: data)
            cachedManifest = decoded
            try? data.write(to: CMZPaths.manifestCache, options: [.atomic])
            return decoded
        } catch {
            if let cached = cachedManifest { return cached }
            if let disk = try? readDiskCache() { cachedManifest = disk; return disk }
            throw CMZError.manifestUnavailable(underlying: error)
        }
    }

    private func readDiskCache() throws -> CMZManifest {
        let url = CMZPaths.manifestCache
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw CMZError.manifestUnavailable(underlying: nil)
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(CMZManifest.self, from: data)
    }

    public func entry(for id: String) async throws -> CMZModelEntry {
        let m = try await manifest()
        guard let e = m.models.first(where: { $0.id == id }) else {
            throw CMZError.unknownModel(id: id)
        }
        return e
    }

    // MARK: - Install state

    public func isInstalled(id: String) -> Bool {
        FileManager.default.fileExists(atPath: CMZPaths.metaFile(modelId: id).path)
    }

    public func installedIds() -> Set<String> {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(
            at: CMZPaths.modelsDir, includingPropertiesForKeys: nil) else { return [] }
        var ids: Set<String> = []
        for url in entries where url.hasDirectoryPath {
            if fm.fileExists(atPath: url.appendingPathComponent(".meta.json").path) {
                ids.insert(url.lastPathComponent)
            }
        }
        return ids
    }

    /// Return the local directory that contains the unpacked model files.
    /// Throws `modelNotInstalled` if the model hasn't been downloaded yet.
    public func localDirectory(for id: String) throws -> URL {
        guard isInstalled(id: id) else { throw CMZError.modelNotInstalled(id: id) }
        return CMZPaths.modelDir(id: id)
    }

    // MARK: - Download

    /// Download (and verify) all files for a model. The returned stream emits
    /// progress snapshots; it finishes when the model is fully installed or
    /// throws on failure.
    ///
    /// Idempotent: calling `download` on an already-installed model emits a
    /// single `.done` progress and returns immediately.
    public nonisolated func download(
        id: String
    ) -> AsyncThrowingStream<CMZDownloadProgress, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let entry = try await self.entry(for: id)
                    try await self.runDownload(entry: entry, continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func runDownload(entry: CMZModelEntry,
                             continuation: AsyncThrowingStream<CMZDownloadProgress, Error>.Continuation) async throws {
        if isInstalled(id: entry.id) {
            continuation.yield(CMZDownloadProgress(modelId: entry.id,
                                                    phase: .done,
                                                    fraction: 1,
                                                    currentFile: nil,
                                                    bytesReceived: entry.totalDownloadBytes,
                                                    bytesExpected: entry.totalDownloadBytes))
            return
        }

        let downloader = downloaders[entry.id] ?? FileDownloader(modelId: entry.id)
        downloaders[entry.id] = downloader

        let totalExpected = entry.totalDownloadBytes
        var completedBytes: Int64 = 0

        for spec in entry.files where !(spec.optional ?? false) {
            let dest = CMZPaths.modelDir(id: entry.id).appendingPathComponent(spec.name)
            let specBytes = Int64(spec.sizeBytes)
            let completedSoFar = completedBytes

            try await withCheckedThrowingContinuation { (fileCont: CheckedContinuation<Void, Error>) in
                Task {
                    do {
                        try await downloader.download(spec: spec, to: dest, callbacks: .init(
                            onProgress: { fraction, received, _ in
                                let overall = Double(completedSoFar) + Double(specBytes) * fraction
                                let frac = totalExpected > 0
                                    ? min(1.0, overall / Double(totalExpected)) : 0
                                continuation.yield(CMZDownloadProgress(
                                    modelId: entry.id,
                                    phase: .downloading,
                                    fraction: frac,
                                    currentFile: spec.name,
                                    bytesReceived: Int64(overall),
                                    bytesExpected: totalExpected))
                            },
                            onPhase: { phase in
                                continuation.yield(CMZDownloadProgress(
                                    modelId: entry.id,
                                    phase: phase,
                                    fraction: phase == .done ? 1 : 0,
                                    currentFile: spec.name,
                                    bytesReceived: completedSoFar + specBytes,
                                    bytesExpected: totalExpected))
                            }
                        ))
                        fileCont.resume()
                    } catch {
                        fileCont.resume(throwing: error)
                    }
                }
            }
            completedBytes += specBytes
        }

        try writeMeta(entry: entry)
        continuation.yield(CMZDownloadProgress(modelId: entry.id,
                                                phase: .done,
                                                fraction: 1,
                                                currentFile: nil,
                                                bytesReceived: totalExpected,
                                                bytesExpected: totalExpected))
    }

    private func writeMeta(entry: CMZModelEntry) throws {
        let payload: [String: Any] = [
            "model_id": entry.id,
            "installed_at": ISO8601DateFormatter().string(from: Date()),
            "files": entry.files.map { ["name": $0.name, "sha256": $0.sha256, "size_bytes": $0.sizeBytes] },
        ]
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
        try data.write(to: CMZPaths.metaFile(modelId: entry.id), options: [.atomic])
    }

    // MARK: - Delete

    public func delete(id: String) throws {
        let dir = CMZPaths.modelDir(id: id)
        try FileManager.default.removeItem(at: dir)
    }
}
