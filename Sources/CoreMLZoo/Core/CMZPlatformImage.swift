import Foundation
import CoreGraphics

#if canImport(UIKit)
import UIKit
public typealias CMZPlatformImage = UIImage
#elseif canImport(AppKit)
import AppKit
public typealias CMZPlatformImage = NSImage
#endif

extension CMZPlatformImage {
    /// Return a `CGImage` with EXIF orientation baked in (iOS only; AppKit
    /// returns the primary rep directly).
    func cmzCGImage() -> CGImage? {
        #if canImport(UIKit)
        if imageOrientation == .up { return cgImage }
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        draw(in: CGRect(origin: .zero, size: size))
        let img = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return img?.cgImage
        #elseif canImport(AppKit)
        var rect = CGRect(origin: .zero, size: size)
        return cgImage(forProposedRect: &rect, context: nil, hints: nil)
        #endif
    }

    static func cmzFromCGImage(_ cg: CGImage) -> CMZPlatformImage {
        #if canImport(UIKit)
        return UIImage(cgImage: cg)
        #elseif canImport(AppKit)
        return NSImage(cgImage: cg, size: CGSize(width: cg.width, height: cg.height))
        #endif
    }
}
