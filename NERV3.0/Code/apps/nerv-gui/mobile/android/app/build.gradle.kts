// Android app module (erratum 205; mobile shell).
//
// The Compose shell hosts the eframe-built WASM bundle from
// `assets/www/index.html` and adds native chrome on top:
//  - Splash screen (NERV logo, branded)
//  - Biometric lock gate (AndroidX Biometric)
//  - Native top bar (logo + chain status + lock)
//  - Native bottom navigation (8 screens mirror the desktop app)
//  - Snackbar host (for native notifications)
//  - JS bridge (Kotlin <-> JS) so the chrome can drive the wallet

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "org.nerv.wallet"
    compileSdk = 34

    defaultConfig {
        applicationId = "org.nerv.wallet"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"
        resourceConfigurations += "en"
        vectorDrawables.useSupportLibrary = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            isShrinkResources = false
            // The release keystore is provisioned by the testnet
            // operator — this app module compiles but signing is a
            // separate concern (see README.md).
            signingConfig = signingConfigs.getByName("debug")
        }
        debug {
            applicationIdSuffix = ".debug"
            isDebuggable = true
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
        freeCompilerArgs += listOf(
            "-opt-in=androidx.compose.material3.ExperimentalMaterial3Api",
            "-opt-in=androidx.compose.foundation.ExperimentalFoundationApi",
            "-opt-in=androidx.compose.animation.ExperimentalAnimationApi",
            "-opt-in=androidx.compose.ui.ExperimentalComposeUiApi",
        )
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.10"
    }

    packaging {
        resources {
            excludes += setOf(
                "/META-INF/{AL2.0,LGPL2.1}",
                "/META-INF/DEPENDENCIES",
                "/META-INF/LICENSE*",
                "/META-INF/NOTICE*",
                "META-INF/*.kotlin_module",
            )
        }
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.02.02")
    implementation(composeBom)

    // Core AndroidX.
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("androidx.activity:activity-compose:1.8.2")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.7.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")

    // Compose.
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.compose.animation:animation")

    // Navigation.
    implementation("androidx.navigation:navigation-compose:2.7.7")

    // Biometrics.
    implementation("androidx.biometric:biometric:1.1.0")

    // Fragment (required for FragmentActivity / BiometricPrompt host).
    implementation("androidx.fragment:fragment-ktx:1.6.2")

    // Encrypted storage for the biometric-unlock state.
    implementation("androidx.security:security-crypto:1.1.0-alpha06")

    // Splash screen (Android 12+).
    implementation("androidx.core:core-splashscreen:1.0.1")

    // Window size class (responsive layouts).
    implementation("androidx.compose.material3:material3-window-size-class")

    // Debug.
    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}