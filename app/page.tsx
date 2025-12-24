export default function Home() {
  return (
    <>
      <div className="hero">
        <h1>NERV</h1>
        <p className="tagline">The private, post-quantum, infinitely scalable blockchain</p>
        <p className="launch">Fair launch June 2028 • No pre-mine • Epics, user-stories, tasks public today</p>
        <div className="buttons">
          <a href="https://github.com/nerv-bit/nerv/blob/main/NERV_Whitepaper_v1.01.pdf" className="btn primary">Read Whitepaper v1.01</a>
          <a href="https://github.com/nerv-bit/nerv" target="_blank" className="btn secondary">GitHub →</a>
        </div>
        {/* Trimmed for scannability */}
        <p className="launch" style={{ marginTop: '2.5rem', maxWidth: '860px', lineHeight: '1.7' }}>
          NERV delivers full privacy by default (&gt;1M TPS via dynamic neural sharding and verifiable embeddings in Halo2/Nova), post-quantum security from genesis, and a self-improving useful-work economy. Fair launch June 2028: zero pre-mine, fully open-source, community-governed. Join us in building the nervous system of the private internet!
        </p>

        {/* New: Architecture Diagram Section */}
        <section className="architecture" style={{ margin: '4rem auto', maxWidth: '900px', textAlign: 'center' }}>
          <h2 style={{ marginBottom: '1.5rem', fontSize: '1.8rem' }}>High-Level Architecture</h2>
          <img 
            src="https://cdn.prod.website-files.com/64c231f464b91d6bd0303294/6711029566dc1475c0a37d98_66f258e47f53e2e2341aaae0_66d16bf1edcb81f15215c5b6_66d16b305dedb7e05c1b0920_diagram-export-8-30-2024-12_18_02-PM.png" 
            alt="NERV Blockchain Architecture: Flow from user wallet through 5-hop TEE mixer, neural state embeddings, dynamic shards, AI-native consensus, to embedding root and VDW receipts"
            style={{
              width: '100%',
              height: 'auto',
              borderRadius: '12px',
              boxShadow: '0 10px 30px rgba(0,0,0,0.2)',
              border: '1px solid #333'
            }}
          />
          <p style={{ marginTop: '1rem', fontSize: '0.9rem', color: '#aaa' }}>
            User → 5-hop TEE Mixer → Dynamic Neural Shards → AI-Native Consensus → 512-byte Embedding Root
          </p>
        </section>
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
          <a href="mailto:namsjeev@gmail.com">Contact → namsjeev@gmail.com</a>
        </div>
      </section>

      <footer>
        <p>© 2025–2028 NERV • All specifications, code, and proofs are MIT/Apache 2.0 or public domain</p>
        <p>No tokens exist yet • No private sales • No foundation treasury</p>
      </footer>
    </>
  )
}
