// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "CoreMLZoo",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
        .macCatalyst(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "CoreMLZoo", targets: ["CoreMLZoo"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "CoreMLZoo",
            path: "Sources/CoreMLZoo",
            resources: [
                .process("Resources"),
            ]
        ),
        .testTarget(
            name: "CoreMLZooTests",
            dependencies: ["CoreMLZoo"],
            path: "Tests/CoreMLZooTests"
        ),
    ]
)
