import XCTest
@testable import CoreMLZoo

final class CoreMLZooTests: XCTestCase {
    func testPublicTypesCompile() {
        // Smoke test: every Request/Session type exposes its expected
        // public surface. No network or models involved.
        _ = BackgroundRemovalRequest()
        _ = DepthRequest(model: .moGe2)
        _ = UpscaleRequest(model: .realESRGAN4x)
        _ = InpaintRequest(model: .lama)
        _ = OpenVocabDetectionRequest()
        _ = FaceEmbeddingRequest()
        _ = Face3DRequest()
        _ = SourceSeparationRequest()
        _ = LowLightEnhanceRequest()
        _ = ImageRestorationRequest(model: .mprnetDeblurring)
        _ = ColorizeRequest()
        _ = MusicTranscriptionRequest()
        _ = ZeroShotClassificationRequest()
    }

    func testCosineSimilarity() {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [1, 0, 0]
        XCTAssertEqual(FaceEmbeddingRequest.cosineSimilarity(a, b), 1, accuracy: 1e-5)
        let c: [Float] = [0, 1, 0]
        XCTAssertEqual(FaceEmbeddingRequest.cosineSimilarity(a, c), 0, accuracy: 1e-5)
    }

    func testComputeUnitsResolution() {
        XCTAssertEqual(CMZComputeUnits.cpuOnly.mlComputeUnits, .cpuOnly)
        XCTAssertEqual(CMZComputeUnits.all.mlComputeUnits, .all)
    }
}
