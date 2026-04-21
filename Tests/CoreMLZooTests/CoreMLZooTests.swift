import XCTest
@testable import CoreMLZoo

final class CoreMLZooTests: XCTestCase {
    func testPublicTypesCompile() {
        // Smoke test: every Request/Session type exposes its expected
        // public surface. No network or models involved.
        _ = BackgroundRemovalRequest()
        _ = DepthRequest(model: .moGe2)
        _ = UpscaleRequest(model: .realESRGAN)
        _ = InpaintRequest(model: .lama)
        _ = OpenVocabDetectionRequest()
        _ = Face3DRequest()
        _ = SourceSeparationRequest()
        _ = ColorizeRequest()
        _ = ZeroShotClassificationRequest()
    }

    func testModelIdsMatchManifest() {
        // Keep this list in sync with the default HF manifest.
        // The Hub App authoritative entries these SDK types map to.
        let expected = [
            "rmbg_1_4", "moge2_vitb_normal_504", "depth_anything_v3_small_504",
            "depth_anything_v3_base_504", "realesrgan", "gfpgan", "lama",
            "yoloworld", "face3d", "demucs", "ddcolor", "siglip",
            "matanyone", "florence2", "kokoro", "openvoice", "stable_audio",
        ]
        for id in expected {
            XCTAssertFalse(id.isEmpty)
        }
    }

    func testComputeUnitsResolution() {
        XCTAssertEqual(CMZComputeUnits.cpuOnly.mlComputeUnits, .cpuOnly)
        XCTAssertEqual(CMZComputeUnits.all.mlComputeUnits, .all)
    }
}
