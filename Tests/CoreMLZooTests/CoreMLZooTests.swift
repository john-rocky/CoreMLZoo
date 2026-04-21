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

    func testFlorenceGroundingBoxParser() {
        // Single box.
        let raw1 = "a red car<loc_100><loc_200><loc_700><loc_800>"
        let rects1 = VisionLanguageSession.parseLocationBoxes(from: raw1)
        XCTAssertEqual(rects1.count, 1)
        XCTAssertEqual(rects1[0].origin.x, 100.0 / 999.0, accuracy: 1e-5)
        XCTAssertEqual(rects1[0].origin.y, 200.0 / 999.0, accuracy: 1e-5)
        XCTAssertEqual(rects1[0].width,    600.0 / 999.0, accuracy: 1e-5)
        XCTAssertEqual(rects1[0].height,   600.0 / 999.0, accuracy: 1e-5)

        // Multiple boxes with swapped coords (parser normalizes).
        let raw2 = "dog<loc_10><loc_20><loc_100><loc_150>cat<loc_500><loc_500><loc_400><loc_450>"
        let rects2 = VisionLanguageSession.parseLocationBoxes(from: raw2)
        XCTAssertEqual(rects2.count, 2)

        // Degenerate boxes are dropped.
        let raw3 = "empty<loc_500><loc_500><loc_500><loc_500>"
        XCTAssertEqual(VisionLanguageSession.parseLocationBoxes(from: raw3).count, 0)

        // No markers → empty.
        XCTAssertEqual(VisionLanguageSession.parseLocationBoxes(from: "plain text").count, 0)

        // Partial group (3 locs) → no rect.
        let raw4 = "partial<loc_10><loc_20><loc_30>"
        XCTAssertEqual(VisionLanguageSession.parseLocationBoxes(from: raw4).count, 0)
    }
}
