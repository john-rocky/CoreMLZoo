import Foundation
import Accelerate

/// sRGB ↔ CIE L*a*b* (D65 illuminant) conversions used by DDColor.
enum LabColor {

    /// Convert sRGB (0..1) triplets → (L, a, b). L in [0, 100], a/b ~[-128, 127].
    static func srgbToLab(r: Float, g: Float, b: Float) -> (Float, Float, Float) {
        func toLinear(_ c: Float) -> Float {
            c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
        }
        let rl = toLinear(r), gl = toLinear(g), bl = toLinear(b)
        var x = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
        var y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
        var z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041
        x /= 0.95047; y /= 1.0; z /= 1.08883
        func f(_ t: Float) -> Float {
            t > 0.008856 ? pow(t, 1.0 / 3.0) : 7.787 * t + 16.0 / 116.0
        }
        let fx = f(x), fy = f(y), fz = f(z)
        return (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
    }

    static func labToSrgb(l: Float, a: Float, b: Float) -> (Float, Float, Float) {
        let fy = (l + 16.0) / 116.0
        let fx = a / 500.0 + fy
        let fz = fy - b / 200.0
        func invF(_ t: Float) -> Float {
            let t3 = t * t * t
            return t3 > 0.008856 ? t3 : (t - 16.0 / 116.0) / 7.787
        }
        let x = invF(fx) * 0.95047
        let y = invF(fy) * 1.0
        let z = invF(fz) * 1.08883
        let r = x *  3.2404542 + y * -1.5371385 + z * -0.4985314
        let g = x * -0.9692660 + y *  1.8760108 + z *  0.0415560
        let bv = x *  0.0556434 + y * -0.2040259 + z *  1.0572252
        func toSRGB(_ c: Float) -> Float {
            let clamped = max(0, min(1, c))
            return clamped <= 0.0031308 ? clamped * 12.92 : 1.055 * pow(clamped, 1.0 / 2.4) - 0.055
        }
        return (toSRGB(r), toSRGB(g), toSRGB(bv))
    }
}
