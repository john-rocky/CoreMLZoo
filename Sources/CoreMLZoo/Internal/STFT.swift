import Foundation
import Accelerate

/// Magnitude spectrogram via Accelerate FFT. Mirrors librosa's default:
/// reflect padding, Hann periodic window, centered frames. Output is
/// `[freqBins][numFrames]` flattened row-major on the freq axis.
enum STFT {

    static func magnitude(samples: UnsafePointer<Float>,
                          count: Int,
                          nFFT: Int,
                          hopLength: Int,
                          winLength: Int) -> [Float] {
        let freqBins = nFFT / 2 + 1
        let padAmount = (nFFT - hopLength) / 2

        var padded = [Float](repeating: 0, count: count + 2 * padAmount)
        for i in 0..<padAmount { padded[i] = samples[min(padAmount - i, count - 1)] }
        memcpy(&padded[padAmount], samples, count * MemoryLayout<Float>.size)
        for i in 0..<padAmount { padded[count + padAmount + i] = samples[max(0, count - 2 - i)] }

        let numFrames = (padded.count - nFFT) / hopLength + 1

        var window = [Float](repeating: 0, count: winLength)
        for i in 0..<winLength {
            window[i] = 0.5 * (1 - cos(2 * .pi * Float(i) / Float(winLength)))
        }

        let log2n = vDSP_Length(log2(Float(nFFT)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return [] }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        var result = [Float](repeating: 0, count: freqBins * numFrames)
        var realp = [Float](repeating: 0, count: nFFT / 2)
        var imagp = [Float](repeating: 0, count: nFFT / 2)

        for frame in 0..<numFrames {
            let start = frame * hopLength
            var windowed = [Float](repeating: 0, count: nFFT)
            for i in 0..<winLength { windowed[i] = padded[start + i] * window[i] }

            realp.withUnsafeMutableBufferPointer { rBuf in
                imagp.withUnsafeMutableBufferPointer { iBuf in
                    var split = DSPSplitComplex(realp: rBuf.baseAddress!, imagp: iBuf.baseAddress!)
                    windowed.withUnsafeBufferPointer { src in
                        src.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: nFFT / 2) { ptr in
                            vDSP_ctoz(ptr, 2, &split, 1, vDSP_Length(nFFT / 2))
                        }
                    }
                    vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))

                    let dc = sqrt(split.realp[0] * split.realp[0] + 1e-6)
                    let ny = sqrt(split.imagp[0] * split.imagp[0] + 1e-6)
                    result[0 * numFrames + frame] = dc
                    for k in 1..<(nFFT / 2) {
                        let r = split.realp[k] / 2
                        let im = split.imagp[k] / 2
                        result[k * numFrames + frame] = sqrt(r * r + im * im + 1e-6)
                    }
                    result[(nFFT / 2) * numFrames + frame] = ny
                }
            }
        }
        return result
    }
}
