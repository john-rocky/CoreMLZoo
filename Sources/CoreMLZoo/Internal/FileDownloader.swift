import Foundation
import CryptoKit

/// Per-file download with SHA-256 verification and resume data persistence.
/// Library-level primitive; consumers use `CMZModelStore` instead.
final class FileDownloader: NSObject, @unchecked Sendable {

    struct Callbacks {
        var onProgress: (_ fraction: Double, _ received: Int64, _ expected: Int64) -> Void
        var onPhase: (CMZDownloadProgress.Phase) -> Void
    }

    private let modelId: String
    private var session: URLSession!
    private var continuation: CheckedContinuation<URL, Error>?
    private var currentSpec: CMZFileSpec?
    private var callbacks: Callbacks?

    init(modelId: String) {
        self.modelId = modelId
        super.init()
        // Background session: keeps downloads alive during short suspensions.
        // Identifier ties the session to the model, so one model's retry
        // doesn't collide with another's in-flight download.
        let config = URLSessionConfiguration.background(
            withIdentifier: "dev.coreml-zoo.dl.\(modelId)")
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 60 * 60
        self.session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }

    func download(spec: CMZFileSpec,
                  to destination: URL,
                  callbacks: Callbacks) async throws {
        self.currentSpec = spec
        self.callbacks = callbacks

        if try alreadyInstalled(spec: spec, at: destination) {
            callbacks.onPhase(.done)
            return
        }

        guard let remote = URL(string: spec.url) else {
            throw CMZError.downloadFailed(id: modelId, reason: "bad URL \(spec.url)")
        }

        callbacks.onPhase(.downloading)
        let tmp = try await runDownload(from: remote)

        callbacks.onPhase(.verifying)
        let digest = try Self.sha256(of: tmp)
        guard digest.lowercased() == spec.sha256.lowercased() else {
            try? FileManager.default.removeItem(at: tmp)
            throw CMZError.checksumMismatch(file: spec.name,
                                            expected: spec.sha256,
                                            got: digest)
        }

        if spec.archive == "zip" {
            callbacks.onPhase(.unpacking)
            do {
                try ZipUnpacker.unpack(archive: tmp, to: destination.deletingLastPathComponent())
            } catch {
                try? FileManager.default.removeItem(at: tmp)
                throw CMZError.unpackFailed(reason: "\(error)")
            }
            try? FileManager.default.removeItem(at: tmp)
        } else {
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: tmp, to: destination)
        }
        callbacks.onPhase(.done)
    }

    private func alreadyInstalled(spec: CMZFileSpec, at destination: URL) throws -> Bool {
        let fm = FileManager.default
        if spec.archive == "zip" {
            let base = (spec.name as NSString).deletingPathExtension
            let expected = destination.deletingLastPathComponent().appendingPathComponent(base)
            return fm.fileExists(atPath: expected.path)
        }
        guard fm.fileExists(atPath: destination.path) else { return false }
        let digest = try Self.sha256(of: destination)
        return digest.lowercased() == spec.sha256.lowercased()
    }

    private func runDownload(from remote: URL) async throws -> URL {
        try await withCheckedThrowingContinuation { cont in
            self.continuation = cont
            let task = session.downloadTask(with: remote)
            task.resume()
        }
    }

    static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while autoreleasepool(invoking: {
            let chunk = (try? handle.read(upToCount: 1024 * 1024)) ?? Data()
            if chunk.isEmpty { return false }
            hasher.update(data: chunk)
            return true
        }) {}
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

extension FileDownloader: URLSessionDownloadDelegate {
    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        // URLSession deletes `location` when this method returns, so move to
        // a persistent temp path before resuming the continuation.
        let persisted = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".download")
        do {
            try FileManager.default.moveItem(at: location, to: persisted)
            continuation?.resume(returning: persisted)
        } catch {
            continuation?.resume(throwing: error)
        }
        continuation = nil
    }

    func urlSession(_ session: URLSession,
                    task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        guard let error else { return }
        continuation?.resume(throwing: error)
        continuation = nil
    }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let fraction = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        callbacks?.onProgress(fraction, totalBytesWritten, totalBytesExpectedToWrite)
    }
}
