// shared/build.gradle.kts — KMP module wrapping the ANE MNIST trainer
plugins {
    kotlin("multiplatform") version "2.1.0"
}

group = "com.ane"
version = "0.1.0"

kotlin {
    // Apple targets (ANE via cinterop)
    macosArm64("apple") {
        compilations.getByName("main") {
            cinterops {
                val anemnist by creating {
                    defFile(project.file("src/nativeInterop/cinterop/anemnist.def"))
                    packageName("anemnist")
                    includeDirs(project.file("../objc-bridge"))
                }
            }
        }
        binaries {
            executable("mnistDemo") {
                entryPoint = "com.ane.mnist.main"
            }
        }
    }

    // JVM target (CPU fallback)
    jvm {
        compilations.all {
            kotlinOptions.jvmTarget = "17"
        }
    }

    sourceSets {
        val commonMain by getting

        val appleMain by getting {
            dependsOn(commonMain)
        }

        val jvmMain by getting {
            dependsOn(commonMain)
        }

        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
            }
        }
    }
}

// Task to compile the ObjC bridge into a static library (needed before Kotlin/Native link)
tasks.register<Exec>("compileObjcBridge") {
    workingDir = file("../objc-bridge")
    commandLine("make", "lib")
    description = "Compile ANEMnist ObjC bridge into libane_mnist.a"
}

// Ensure the ObjC bridge is compiled before Kotlin/Native links
tasks.matching { it.name.contains("linkMnistDemo") || it.name.contains("CInterop") }.configureEach {
    dependsOn("compileObjcBridge")
}
