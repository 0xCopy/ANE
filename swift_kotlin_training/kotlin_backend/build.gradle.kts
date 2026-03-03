// kotlin_backend/build.gradle.kts
plugins {
    kotlin("jvm") version "1.9.25"
    id("com.google.devtools.ksp") version "1.9.25-1.0.15"
    id("com.github.jengelman.shadow") version "8.1.1"
}

group = "com.example"
version = "1.0.0"

repositories {
    mavenCentral()
    maven { url = uri("https://oss.sonatype.org/content/repositories/snapshots") }
    maven { url = uri("https://maven.pytorch.org/repository/releases") }
}

dependencies {
    implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8:1.9.25")

    // PyTorch Kotlin bindings
    implementation("org.pytorch:pytorch_android:2.1.0")
    implementation("org.pytorch:pytorch_android_torchvision:2.1.0")

    // Torch Kotlin (experimental)
    implementation("org.torch:torch:2.1.0")
    implementation("org.torch:torch-kotlin:2.1.0")

    // JSON processing
    implementation("com.google.code.gson:gson:2.10.1")

    // Testing
    testImplementation("org.jetbrains.kotlin:kotlin-test:1.9.25")
}

kotlin {
    jvmToolchain(17)
}

tasks.named<Jar>("jar") {
    duplicatesStrategy = DuplicatesStrategy.INCLUDE
}

tasks.named<ShadowJar>("shadowJar") {
    archiveBaseName.set("kotlin-torch-backend")
    archiveClassifier.set("")
    manifest {
        attributes['Main-Class'] = 'com.example.KotlinTorchTraining'
    }
}
