import CoreML

/// User-facing compute unit preference. Mapped to `MLComputeUnits` at load time.
///
/// `.auto` defers to the per-model default recommended by the manifest
/// (which encodes hard-learned rules like "MatAnyone read/read_first must be
/// `.cpuOnly` to work around the iOS GPU singleton-dim slice bug"). Pick an
/// explicit case only when you need to override that default.
public enum CMZComputeUnits: Sendable, Hashable {
    case auto
    case cpuOnly
    case cpuAndGPU
    case cpuAndNeuralEngine
    case all

    var mlComputeUnits: MLComputeUnits {
        switch self {
        case .auto, .all:       return .all
        case .cpuOnly:          return .cpuOnly
        case .cpuAndGPU:        return .cpuAndGPU
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        }
    }

    static func resolve(_ requested: CMZComputeUnits, manifestHint: String?) -> MLComputeUnits {
        if case .auto = requested {
            switch manifestHint {
            case "cpu_only":            return .cpuOnly
            case "cpu_and_gpu":         return .cpuAndGPU
            case "cpu_and_ne":          return .cpuAndNeuralEngine
            case "all", nil:            return .all
            default:                    return .all
            }
        }
        return requested.mlComputeUnits
    }
}
