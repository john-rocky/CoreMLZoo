import Foundation
import CoreML

/// A single-shot inference. Stateless — construct, `perform`, done.
///
/// Compared to `VNRequest`: no handler, no delegate, no mutable observation
/// state. `perform` is async and returns the result directly.
public protocol CMZRequest {
    associatedtype Input
    associatedtype Output

    /// Model id this request depends on. Must be installed via
    /// `CMZModelStore.shared.download(id:)` before `perform` is called.
    var modelId: String { get }

    func perform(on input: Input) async throws -> Output
}

public extension CMZRequest {
    /// Convenience: ensure the underlying model is installed before running.
    /// Streams download progress if a download is needed.
    func prepareAndPerform(
        on input: Input,
        progress: (@Sendable (CMZDownloadProgress) -> Void)? = nil
    ) async throws -> Output {
        if !(await CMZModelStore.shared.isInstalled(id: modelId)) {
            for try await p in CMZModelStore.shared.download(id: modelId) {
                progress?(p)
            }
        }
        return try await perform(on: input)
    }
}

/// A stateful pipeline. Loads models once at init, keeps them alive, exposes
/// per-call mutating methods. Used for seq2seq decoding (Florence-2),
/// feedback-loop video (MatAnyone), TTS (Kokoro), etc.
public protocol CMZSession: AnyObject {
    var modelIds: [String] { get }
}
