import Foundation

/// Mirror of the manifest schema used by CoreML-Models Hub App.
/// Source of truth: `https://huggingface.co/mlboydaisuke/coreml-zoo/resolve/main/models.json`
///
/// Only the fields the SDK actually reads are modelled here; unknown fields are
/// preserved as `AnyCodable` so the manifest can evolve without breaking the SDK.
public struct CMZManifest: Codable, Sendable {
    public let manifestVersion: Int
    public let updatedAt: String
    public let minAppVersion: String
    public let categories: [CMZCategory]
    public let models: [CMZModelEntry]

    enum CodingKeys: String, CodingKey {
        case manifestVersion = "manifest_version"
        case updatedAt = "updated_at"
        case minAppVersion = "min_app_version"
        case categories
        case models
    }
}

public struct CMZCategory: Codable, Sendable, Hashable, Identifiable {
    public let id: String
    public let name: String
    public let icon: String
    public let order: Int
}

public struct CMZModelEntry: Codable, Sendable, Hashable, Identifiable {
    public let id: String
    public let name: String
    public let subtitle: String?
    public let categoryId: String
    public let files: [CMZFileSpec]
    public let requirements: CMZRequirements
    public let license: CMZLicense
    public let upstream: CMZUpstream
    public let demo: CMZDemo?
    public let hfRepoUrl: String?

    enum CodingKeys: String, CodingKey {
        case id, name, subtitle
        case categoryId = "category_id"
        case files, requirements, license, upstream, demo
        case hfRepoUrl = "hf_repo_url"
    }

    public var totalDownloadBytes: Int64 {
        files.reduce(0) { $0 + ($1.optional == true ? 0 : Int64($1.sizeBytes)) }
    }

    public static func == (lhs: CMZModelEntry, rhs: CMZModelEntry) -> Bool { lhs.id == rhs.id }
    public func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

public struct CMZFileSpec: Codable, Sendable, Hashable, Identifiable {
    public var id: String { name }
    public let name: String
    public let url: String
    public let archive: String?      // "zip" or nil
    public let sizeBytes: Int
    public let sha256: String
    public let computeUnits: String? // "cpu_only" | "cpu_and_gpu" | "cpu_and_ne" | "all"
    public let optional: Bool?
    public let kind: String?         // "mlpackage" | "tokenizer" | "vocab" | ...

    enum CodingKeys: String, CodingKey {
        case name, url, archive
        case sizeBytes = "size_bytes"
        case sha256
        case computeUnits = "compute_units"
        case optional, kind
    }
}

public struct CMZRequirements: Codable, Sendable, Hashable {
    public let minIos: String
    public let minRamMb: Int
    public let deviceCapabilities: [String]?

    enum CodingKeys: String, CodingKey {
        case minIos = "min_ios"
        case minRamMb = "min_ram_mb"
        case deviceCapabilities = "device_capabilities"
    }
}

public struct CMZLicense: Codable, Sendable, Hashable {
    public let name: String
    public let url: String
}

public struct CMZUpstream: Codable, Sendable, Hashable {
    public let name: String
    public let url: String
    public let year: Int?
}

public struct CMZDemo: Codable, Sendable, Hashable {
    public let template: String
    // We deliberately don't decode the `config` dictionary — Request types know
    // their own config statically, manifest config is UI-layer state for the
    // Hub App.
}
