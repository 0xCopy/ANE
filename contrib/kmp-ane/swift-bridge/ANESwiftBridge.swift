// ANESwiftBridge.swift — Swift-friendly wrapper around ANEBridge C API
// This provides a clean Swift interface that Kotlin/Native can also consume
// via the generated Objective-C header.

import Foundation

/// Status codes mirroring ANEBridgeStatus
@objc public enum ANEStatus: Int32 {
    case ok             =  0
    case initFailed     = -1
    case compileFailed  = -2
    case loadFailed     = -3
    case evalFailed     = -4
    case nullKernel     = -5
    case indexOOB       = -6
    case allocFailed    = -7
}

/// A compiled ANE kernel handle, wrapping the C ANEBridgeKernel pointer.
@objc public class ANEKernelHandle: NSObject {
    let raw: OpaquePointer  // ANEBridgeKernel*

    init(_ ptr: OpaquePointer) {
        self.raw = ptr
    }

    deinit {
        ane_bridge_free_kernel(UnsafeMutablePointer(raw))
    }
}

/// Swift wrapper around the ANE bridge for compiling and evaluating MIL programs.
@objc public class ANEEngine: NSObject {

    @objc public static let shared = ANEEngine()

    @objc public private(set) var isAvailable: Bool = false

    private override init() {
        super.init()
        let status = ane_bridge_init()
        isAvailable = (status == ANE_OK)
    }

    /// Compile a baked-weight conv kernel.
    /// - Parameters:
    ///   - weights: Float32 weight array [out_ch × in_ch], row-major
    ///   - inCh: input channels
    ///   - outCh: output channels
    ///   - spatial: spatial dimension (batch size for MNIST = number of samples)
    /// - Returns: kernel handle or nil on failure
    @objc public func compileConv(weights: UnsafePointer<Float>,
                                   inCh: Int32,
                                   outCh: Int32,
                                   spatial: Int32) -> ANEKernelHandle? {
        var ptr: UnsafeMutablePointer<ANEBridgeKernel>?
        let status = ane_bridge_compile_conv(weights, inCh, outCh, spatial, &ptr)
        guard status == ANE_OK, let p = ptr else { return nil }
        return ANEKernelHandle(OpaquePointer(p))
    }

    /// Evaluate a conv kernel: x[S, inDim] → y[S, outDim]
    /// Handles row-major ↔ channel-first transpose internally.
    @objc public func convEval(kernel: ANEKernelHandle,
                                x: UnsafePointer<Float>,
                                y: UnsafeMutablePointer<Float>,
                                S: Int32,
                                inDim: Int32,
                                outDim: Int32) -> ANEStatus {
        let raw = ane_bridge_conv_eval(
            UnsafeMutablePointer(kernel.raw), x, y, S, inDim, outDim)
        return ANEStatus(rawValue: raw) ?? .evalFailed
    }

    /// Free a kernel (also called automatically on deinit).
    @objc public func freeKernel(_ kernel: ANEKernelHandle) {
        ane_bridge_free_kernel(UnsafeMutablePointer(kernel.raw))
    }

    /// Compile arbitrary MIL text with optional weight blob.
    @objc public func compile(milText: String,
                               weightData: Data?,
                               nInputs: Int32,
                               inputBytes: [Int],
                               nOutputs: Int32,
                               outputBytes: [Int]) -> ANEKernelHandle? {
        let milCStr = milText.utf8CString
        var ptr: UnsafeMutablePointer<ANEBridgeKernel>?
        let inSizes = inputBytes.map { size_t($0) }
        let outSizes = outputBytes.map { size_t($0) }

        let status: Int32
        if let wd = weightData {
            status = wd.withUnsafeBytes { wPtr in
                inSizes.withUnsafeBufferPointer { iPtr in
                    outSizes.withUnsafeBufferPointer { oPtr in
                        milCStr.withUnsafeBufferPointer { mPtr in
                            ane_bridge_compile(
                                mPtr.baseAddress!.withMemoryRebound(to: CChar.self, capacity: mPtr.count) { $0 },
                                milCStr.count - 1,
                                wPtr.baseAddress, wd.count,
                                nInputs, iPtr.baseAddress!,
                                nOutputs, oPtr.baseAddress!,
                                &ptr)
                        }
                    }
                }
            }
        } else {
            status = inSizes.withUnsafeBufferPointer { iPtr in
                outSizes.withUnsafeBufferPointer { oPtr in
                    milCStr.withUnsafeBufferPointer { mPtr in
                        ane_bridge_compile(
                            mPtr.baseAddress!.withMemoryRebound(to: CChar.self, capacity: mPtr.count) { $0 },
                            milCStr.count - 1,
                            nil, 0,
                            nInputs, iPtr.baseAddress!,
                            nOutputs, oPtr.baseAddress!,
                            &ptr)
                    }
                }
            }
        }

        guard status == ANE_OK, let p = ptr else { return nil }
        return ANEKernelHandle(OpaquePointer(p))
    }
}
