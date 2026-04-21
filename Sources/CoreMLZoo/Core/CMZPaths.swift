import Foundation

/// On-disk layout used by the SDK. Deliberately distinct from the Hub App's
/// directory (`coreml-models/`) so that installing an SDK-using app doesn't
/// interfere with a Hub App install on the same device.
///
/// Layout:
///   Application Support/coreml-zoo/
///   ├── manifest.json                  (cached copy of models.json)
///   └── models/
///       └── {model_id}/
///           ├── <file 1>
///           └── .meta.json             (install marker: present ⇒ verified)
enum CMZPaths {
    static let defaultManifestURL =
        URL(string: "https://huggingface.co/mlboydaisuke/coreml-zoo/resolve/main/models.json")!

    static var root: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory,
                                            in: .userDomainMask).first!
        let root = base.appendingPathComponent("coreml-zoo", isDirectory: true)
        ensure(root)
        return root
    }

    static var manifestCache: URL { root.appendingPathComponent("manifest.json") }
    static var manifestEtag: URL { root.appendingPathComponent("manifest.etag") }

    static var modelsDir: URL {
        let url = root.appendingPathComponent("models", isDirectory: true)
        ensure(url)
        return url
    }

    static func modelDir(id: String) -> URL {
        let url = modelsDir.appendingPathComponent(id, isDirectory: true)
        ensure(url)
        return url
    }

    static func metaFile(modelId: String) -> URL {
        modelDir(id: modelId).appendingPathComponent(".meta.json")
    }

    private static func ensure(_ url: URL) {
        let fm = FileManager.default
        if !fm.fileExists(atPath: url.path) {
            try? fm.createDirectory(at: url, withIntermediateDirectories: true)
            var mutable = url
            var rv = URLResourceValues()
            rv.isExcludedFromBackup = true
            try? mutable.setResourceValues(rv)
        }
    }
}
