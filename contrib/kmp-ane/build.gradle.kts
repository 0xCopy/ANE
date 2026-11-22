// build.gradle.kts — KMP project: MNIST training on Apple Neural Engine
//
// Root module: Kotlin/Native MNIST trainer with KotlinGrad-style AD → ANE dispatch
//   src/commonMain/  — Platform-agnostic autograd engine, model, data types
//   src/macosMain/   — Kotlin/Native actual impls (file I/O, ANE cinterop, entry point)
//
// Legacy subproject :shared — Alternative approach with expect/actual MnistTrainer class
//
// Quick start:
//   make bridge                   — compile ObjC bridge into build/libanebridge.a
//   python3 prepare_mnist.py      — download MNIST dataset to data/
//   ./gradlew linkMnistTrainerDebugExecutableMacosArm64
//   ./build/bin/macosArm64/mnistTrainerDebugExecutable/mnist_trainer.kexe
//
// Or the combined command:
//   make all && make run

plugins {
    kotlin("multiplatform") version "2.1.0"
}

group = "dev.ane"
version = "0.1.0"

repositories {
    mavenCentral()
}

kotlin {
    // Primary target: Apple Silicon with ANE
    macosArm64 {
        compilations.getByName("main") {
            cinterops {
                val anebridge by creating {
                    defFile(project.file("src/nativeInterop/cinterop/anebridge.def"))
                    packageName("dev.ane.cinterop")
                    includeDirs(project.file("objc-bridge"))
                }
            }
        }
        binaries {
            executable("mnistTrainer") {
                entryPoint = "dev.ane.mnist.main"
                linkerOpts(
                    "-L${project.projectDir}/build",
                    "-lanebridge",
                    "-framework", "Foundation",
                    "-framework", "IOSurface",
                    "-ldl",
                    "-lobjc"
                )
            }
            executable("mnistKotlinGrad") {
                entryPoint = "dev.ane.mnist.kotlingrad.main"
                linkerOpts(
                    "-L${project.projectDir}/build",
                    "-lanebridge",
                    "-framework", "Foundation",
                    "-framework", "IOSurface",
                    "-ldl",
                    "-lobjc"
                )
            }
        }
    }

    // Optional: x86_64 for Rosetta testing (CPU-only, ANE not available)
    macosX64 {
        compilations.getByName("main") {
            cinterops {
                val anebridge by creating {
                    defFile(project.file("src/nativeInterop/cinterop/anebridge.def"))
                    packageName("dev.ane.cinterop")
                    includeDirs(project.file("objc-bridge"))
                }
            }
        }
        binaries {
            executable("mnistTrainer") {
                entryPoint = "dev.ane.mnist.main"
                linkerOpts(
                    "-L${project.projectDir}/build",
                    "-lanebridge",
                    "-framework", "Foundation",
                    "-framework", "IOSurface",
                    "-ldl",
                    "-lobjc"
                )
            }
            executable("mnistKotlinGrad") {
                entryPoint = "dev.ane.mnist.kotlingrad.main"
                linkerOpts(
                    "-L${project.projectDir}/build",
                    "-lanebridge",
                    "-framework", "Foundation",
                    "-framework", "IOSurface",
                    "-ldl",
                    "-lobjc"
                )
            }
        }
    }

    sourceSets {
        val commonMain by getting

        val macosArm64Main by getting {
            dependsOn(commonMain)
        }

        val macosX64Main by getting {
            dependsOn(commonMain)
        }

        // Shared macOS source set for both arm64 and x64
        val macosMain by creating {
            dependsOn(commonMain)
            macosArm64Main.dependsOn(this)
            macosX64Main.dependsOn(this)
        }
    }
}

// Ensure the ObjC bridge static lib is compiled before Kotlin/Native links
tasks.register<Exec>("compileAneBridge") {
    workingDir = file(".")
    commandLine("make", "-C", ".", "build/libanebridge.a")
    description = "Compile ANEBridge.m into build/libanebridge.a"
}

tasks.matching {
    it.name.contains("linkMnistTrainer") || it.name.contains("linkMnistKotlinGrad") || it.name.contains("CInterop")
}.configureEach {
    dependsOn("compileAneBridge")
}
