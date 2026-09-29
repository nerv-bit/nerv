// Application class (erratum 205).
//
// The application instance is the only place that holds the
// encrypted prefs handle and a global static for the long-lived
// context. It's deliberately minimal — wallet logic lives in Rust.

package org.nerv.wallet

import android.app.Application

class NervApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        instance = this
    }

    companion object {
        lateinit var instance: NervApplication
            private set
    }
}