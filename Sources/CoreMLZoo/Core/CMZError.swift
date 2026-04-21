import Foundation

public enum CMZError: LocalizedError {
    case modelNotInstalled(id: String)
    case unknownModel(id: String)
    case manifestUnavailable(underlying: Error?)
    case downloadFailed(id: String, reason: String)
    case checksumMismatch(file: String, expected: String, got: String)
    case unpackFailed(reason: String)
    case invalidInput(reason: String)
    case inferenceFailed(reason: String)
    case unsupportedPlatform(reason: String)

    public var errorDescription: String? {
        switch self {
        case .modelNotInstalled(let id):
            return "Model '\(id)' is not installed. Call CMZModelStore.shared.download(_:) first."
        case .unknownModel(let id):
            return "Unknown model id '\(id)'."
        case .manifestUnavailable(let err):
            return "Model manifest unavailable\(err.map { ": \($0.localizedDescription)" } ?? ".")"
        case .downloadFailed(let id, let reason):
            return "Download for '\(id)' failed: \(reason)"
        case .checksumMismatch(let file, let expected, let got):
            return "SHA-256 mismatch for \(file): expected \(expected.prefix(12))…, got \(got.prefix(12))…"
        case .unpackFailed(let reason):
            return "Archive extraction failed: \(reason)"
        case .invalidInput(let reason):
            return "Invalid input: \(reason)"
        case .inferenceFailed(let reason):
            return "Inference failed: \(reason)"
        case .unsupportedPlatform(let reason):
            return "Unsupported platform: \(reason)"
        }
    }
}
