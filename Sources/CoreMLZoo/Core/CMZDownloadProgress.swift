import Foundation

/// Progress snapshot emitted by `CMZModelStore.download(_:)`.
public struct CMZDownloadProgress: Sendable {
    public enum Phase: Sendable { case downloading, unpacking, verifying, done }

    public let modelId: String
    public let phase: Phase
    /// Overall 0…1 across all files in the model.
    public let fraction: Double
    /// Currently-active file name (nil during `.done`).
    public let currentFile: String?
    public let bytesReceived: Int64
    public let bytesExpected: Int64
}
