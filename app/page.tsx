export default function Home() {
  return (
    <>
      <div className="hero">
        <h1>NERV</h1>
        <p className="tagline">The private, post-quantum, infinitely scalable blockchain</p>
        <p className="launch">Fair launch June 2028 • No pre-mine • All code public today</p>

        <div className="buttons">
          <a href="/NERV_Whitepaper_v1.01.pdf" className="btn primary">Read Whitepaper v1.01</a>
          <a href="https://github.com/nerv-bit" target="_blank" className="btn secondary">GitHub →</a>
        </div>
      </div>

      <section className="promise">
        <div className="grid">
          <div>Privacy by default</div>
          <div>1 M+ TPS target</div>
          <div>Useful-work economy</div>
          <div>Quantum-secure from day 0</div>
        </div>
      </section>

      <section className="timeline">
        <h2>Road to Mainnet (100% transparent)</h2>
        <div className="timeline-items">
          <div><span>Dec 2025</span> Whitepaper + all code & proofs public</div>
          <div><span>Q1 2026</span> First multi-vendor TEE mixer testnet</div>
          <div><span>Q2 2026</span> Aurora public testnet (real metrics published)</div>
          <div><span>June 2028</span> Fair mainnet launch – zero pre-mine</div>
        </div>
      </section>

      <section className="links">
        <h2>Join the nervous system</h2>
        <div className="link-grid">
          <a href="https://github.com/nerv-bit" target="_blank">GitHub Organization (10+ repos)</a>
          <a href="https://github.com/nerv-bit/formal" target="_blank">Lean 4 Formal Proofs (live)</a>
          <a href="https://github.com/nerv-bit/circuits" target="_blank">Halo2 Circuits</a>
          <a href="https://github.com/nerv-bit/simulations" target="_blank">10 000-node Simulator</a>
          <a href="mailto:hello@nerv.network">Contact → hello@nerv.network</a>
        </div>
      </section>

      <footer>
        <p>© 2025–2028 NERV • All specifications, code, and proofs are MIT/Apache 2.0 or public domain</p>
        <p>No tokens exist yet • No private sales • No foundation treasury</p>
      </footer>
    </>
  )
}
