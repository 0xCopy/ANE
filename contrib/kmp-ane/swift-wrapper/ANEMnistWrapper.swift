// ANEMnistWrapper.swift — Swift wrapper over the C ANEMnist bridge
// Provides a Swift-native API for MNIST training on ANE.
// Used by: SwiftUI apps, KMP Swift interop, standalone Swift scripts.

import Foundation

/// Result of a single training step.
public struct MnistStepResult {
    public let loss: Float
    public let accuracy: Float
    public let step: Int
    public let epoch: Int
    public let epochDone: Bool
    public let compileCount: Int
    public let budgetExhausted: Bool
}

/// Configuration for the MNIST trainer.
public struct MnistConfiguration {
    public var inputDim: Int
    public var hiddenDim: Int
    public var outputDim: Int
    public var batchSize: Int
    public var learningRate: Float
    public var momentum: Float
    public var accumSteps: Int
    public var maxCompiles: Int

    public init(inputDim: Int = 784,
                hiddenDim: Int = 128,
                outputDim: Int = 10,
                batchSize: Int = 128,
                learningRate: Float = 0.01,
                momentum: Float = 0.9,
                accumSteps: Int = 50,
                maxCompiles: Int = 118) {
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.outputDim = outputDim
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.momentum = momentum
        self.accumSteps = accumSteps
        self.maxCompiles = maxCompiles
    }

    internal var cConfig: ANEMnistConfig {
        ANEMnistConfig(
            input_dim: Int32(inputDim),
            hidden_dim: Int32(hiddenDim),
            output_dim: Int32(outputDim),
            batch_size: Int32(batchSize),
            learning_rate: learningRate,
            momentum: momentum,
            accum_steps: Int32(accumSteps),
            max_compiles: Int32(maxCompiles)
        )
    }
}

/// Swift wrapper around the C ANEMnist trainer.
/// Provides MNIST training powered by Apple Neural Engine conv kernels
/// with baked fp16 weights and gradient accumulation.
///
/// Usage:
/// ```swift
/// let trainer = try ANEMnistSwiftTrainer()
/// try trainer.loadData(dataDir: "data")
/// try trainer.compile()
///
/// for _ in 0..<10 {
///     let bpe = trainer.batchesPerEpoch
///     for _ in 0..<bpe {
///         let result = trainer.trainStep()
///         // result.loss, result.accuracy, result.epochDone
///     }
///     let testAcc = trainer.evaluate()
///     print("Test accuracy: \(testAcc)%")
/// }
/// ```
public class ANEMnistSwiftTrainer {
    private var handle: OpaquePointer?

    public let config: MnistConfiguration

    /// Number of training samples loaded (0 until loadData is called).
    public var trainCount: Int {
        guard let h = handle else { return 0 }
        return Int(ane_mnist_train_count(h))
    }

    /// Number of test samples loaded.
    public var testCount: Int {
        guard let h = handle else { return 0 }
        return Int(ane_mnist_test_count(h))
    }

    /// Number of batches per training epoch.
    public var batchesPerEpoch: Int {
        guard let h = handle else { return 0 }
        return Int(ane_mnist_batches_per_epoch(h))
    }

    /// Current loss from the last training step.
    public var currentLoss: Float {
        guard let h = handle else { return 0 }
        return ane_mnist_current_loss(h)
    }

    /// Global step counter.
    public var currentStep: Int {
        guard let h = handle else { return 0 }
        return Int(ane_mnist_current_step(h))
    }

    /// Number of ANE compiles used so far.
    public var compileCount: Int {
        guard let h = handle else { return 0 }
        return Int(ane_mnist_compile_count(h))
    }

    /// Create a trainer with the given configuration.
    public init(config: MnistConfiguration = MnistConfiguration()) throws {
        self.config = config
        var cc = config.cConfig
        guard let h = ane_mnist_create(cc) else {
            throw ANEMnistError.initFailed
        }
        self.handle = h
    }

    /// Load MNIST data from the given directory (expects IDX files).
    public func loadData(dataDir: String) throws {
        guard let h = handle else { throw ANEMnistError.notInitialized }
        let result = ane_mnist_load_data(
            h,
            "\(dataDir)/train-images-idx3-ubyte",
            "\(dataDir)/train-labels-idx1-ubyte",
            "\(dataDir)/t10k-images-idx3-ubyte",
            "\(dataDir)/t10k-labels-idx1-ubyte"
        )
        if result != 0 {
            throw ANEMnistError.dataLoadFailed(code: Int(result))
        }
    }

    /// Load MNIST data from explicit file paths.
    public func loadData(trainImages: String, trainLabels: String,
                         testImages: String, testLabels: String) throws {
        guard let h = handle else { throw ANEMnistError.notInitialized }
        let result = ane_mnist_load_data(h, trainImages, trainLabels,
                                         testImages, testLabels)
        if result != 0 {
            throw ANEMnistError.dataLoadFailed(code: Int(result))
        }
    }

    /// Compile ANE conv kernels with baked weights. Call once after loadData, before training.
    public func compile() throws {
        guard let h = handle else { throw ANEMnistError.notInitialized }
        let result = ane_mnist_compile(h)
        if result != 0 {
            throw ANEMnistError.compileFailed(code: Int(result))
        }
    }

    /// Run one training step (forward + backward + SGD).
    @discardableResult
    public func trainStep() -> MnistStepResult {
        guard let h = handle else {
            return MnistStepResult(loss: 0, accuracy: 0, step: 0,
                                   epoch: 0, epochDone: false,
                                   compileCount: 0, budgetExhausted: true)
        }
        let r = ane_mnist_train_step(h)
        return MnistStepResult(
            loss: r.loss,
            accuracy: r.accuracy,
            step: Int(r.step),
            epoch: Int(r.epoch),
            epochDone: r.epoch_done,
            compileCount: Int(r.compile_count),
            budgetExhausted: r.budget_exhausted
        )
    }

    /// Evaluate accuracy on the test set. Returns percentage [0, 100].
    public func evaluate() -> Float {
        guard let h = handle else { return 0 }
        return ane_mnist_evaluate(h)
    }

    /// Train for a number of epochs, calling the progress callback periodically.
    public func train(epochs: Int,
                      logEvery: Int = 100,
                      progress: ((MnistStepResult) -> Void)? = nil,
                      epochEnd: ((Int, Float) -> Void)? = nil) {
        for _ in 0..<epochs {
            let bpe = batchesPerEpoch
            for step in 0..<bpe {
                let result = trainStep()
                if (step + 1) % logEvery == 0 {
                    progress?(result)
                }
                if result.budgetExhausted {
                    return // ANE compile budget exhausted
                }
                if result.epochDone {
                    let testAcc = evaluate()
                    epochEnd?(result.epoch, testAcc)
                }
            }
        }
    }

    deinit {
        if let h = handle {
            ane_mnist_free(h)
        }
    }
}

/// Errors from the ANE MNIST trainer.
public enum ANEMnistError: Error, CustomStringConvertible {
    case initFailed
    case notInitialized
    case dataLoadFailed(code: Int)
    case compileFailed(code: Int)

    public var description: String {
        switch self {
        case .initFailed: return "Failed to create ANE MNIST trainer"
        case .notInitialized: return "Trainer not initialized"
        case .dataLoadFailed(let c): return "MNIST data load failed (code \(c))"
        case .compileFailed(let c): return "ANE kernel compile failed (code \(c))"
        }
    }
}
