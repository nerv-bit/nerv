"use client";

import { useState } from 'react';
import Image from 'next/image';
import NeuralBackground from './NeuralBackground';

export default function Home() {
  const [heroHovered, setHeroHovered] = useState(false);

  return (
    <div className="relative min-h-screen bg-black text-white overflow-x-hidden">
      {/* Neural Background */}
      <div className="absolute inset-0 z-0">
        <NeuralBackground isActive={heroHovered} />
      </div>

      {/* HERO SECTION */}
      <section
        className="relative pt-10 md:pt-0 md:absolute md:-top-24 md:left-1/2 md:-translate-x-1/2 w-full max-w-5xl px-4 text-center"
        onMouseEnter={() => setHeroHovered(true)}
        onMouseLeave={() => setHeroHovered(false)}
      >
        {/* Cryptographic Icons */}
        <div className="crypto-icon absolute left-[5%] top-1/4 text-3xl opacity-0">🔐</div>
        <div className="crypto-icon absolute right-[7%] top-1/3 text-2xl opacity-0">{`</>`}</div>
        <div className="crypto-icon absolute left-[10%] bottom-1/3 text-xl opacity-0">{`{ }`}</div>
        <div className="crypto-icon absolute right-[10%] bottom-1/4 text-3xl opacity-0">🛡️</div>

        <div className="glass-hero mt-0 pt-4 pb-8">
          <div className="relative mx-auto mb-4 w-full max-w-[220px] sm:max-w-[280px] md:max-w-[420px]">
            <Image
              src="/NERV Logo.png"
              alt="NERV — Neural Encrypted Virtual Relay"
              width={840}
              height={320}
              priority
              className="w-full h-auto transition-all duration-500 ease-out"
              style={{
                animation: heroHovered ? 'nervGlowPulse 3.2s ease-in-out infinite' : 'none',
                filter: heroHovered
                  ? 'drop-shadow(0 0 40px rgba(0, 255, 255, 0.5)) drop-shadow(0 0 90px rgba(168, 85, 247, 0.3))'
                  : 'drop-shadow(0 0 20px rgba(0, 255, 255, 0.25))',
              }}
            />
          </div>

          <div className="h-1 w-24 bg-gradient-to-r from-cyan-500 to-purple-500 mx-auto my-6 rounded-full"></div>

          <p className="tagline text-2xl md:text-3xl mb-4 opacity-90">
            Privacy by Default. Post-Quantum from Genesis. Infinite Scalability.
          </p>

          <p className="launch text-lg md:text-xl mb-12 opacity-80">
            Fair launch October 2027 • No pre-mine • All code public today
          </p>

          <div className="buttons flex flex-col sm:flex-row gap-6 justify-center mb-8">
            <a
              href="https://github.com/nerv-bit/nerv/blob/main/NERV2.0/NERV%20Whitepaper%20V2.0.pdf"
              className="btn primary bg-cyan-500 hover:bg-cyan-400 text-black font-semibold py-3 px-8 rounded-lg transition hover:scale-105"
            >
              Read Whitepaper v2.0
            </a>
            <a
              href="https://github.com/nerv-bit/nerv/tree/main/NERV2.0"
              target="_blank"
              className="btn secondary border border-purple-500 text-white font-semibold py-3 px-8 rounded-lg transition hover:scale-105"
            >
              GitHub →
            </a>
          </div>

          <p className="launch text-lg md:text-xl mb-12 opacity-80">
            <span className="gradient-text font-semibold">NERV</span> delivers full privacy by default via Sphinx 5-hop mixnet (&gt;1M TPS via dynamic neural sharding and verifiable embeddings in Halo2/Nova), post-quantum security from genesis, and a self-improving useful-work economy. Fully open-source, community-governed. Join us in building the <span className="gradient-text font-semibold">nervous system of the private internet</span>!
          </p>
        </div>
      </section>

      {/* MAIN CONTENT */}
      <div className="relative z-10 pt-8 md:pt-[520px]">

        {/* ═══════════════════════════════════════════ */}
        {/* CORE INNOVATIONS SECTION — FIRST            */}
        {/* ═══════════════════════════════════════════ */}
        <section className="promise py-12 text-center max-w-5xl mx-auto px-4">
          <h2 className="text-4xl font-bold mb-8">Core Pillars of NERV</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="neural-card p-6">
              <div className="text-4xl mb-4">⚡</div>
              <h3 className="text-xl font-bold mb-3 gradient-text">Real-Time Self-Evolution</h3>
              <p className="opacity-80">
                Every block, the network learns. Validators earn rewards by submitting gradients that reduce Huber loss — no 30-day federated learning rounds, no Shapley complexity. The network&apos;s Neural Weight Oscillator (NWO) adapts to shifting transaction patterns in real-time.
              </p>
            </div>
            <div className="neural-card p-6">
              <div className="text-4xl mb-4">🧠</div>
              <h3 className="text-xl font-bold mb-3 gradient-text">Neural State Embeddings</h3>
              <p className="opacity-80">Replace Merkle trees with 512-byte homomorphic embeddings. State updates are O(1) additions, not O(log n) tree traversals. Dynamic neural sharding splits and merges like cells — no manual rebalancing, no theoretical TPS ceiling.</p>
            </div>
            <div className="neural-card p-6">
              <div className="text-4xl mb-4">🔐</div>
              <h3 className="text-xl font-bold mb-3 gradient-text">Post-Quantum Cryptography</h3>
              <p className="opacity-80">CRYSTALS-Dilithium-3 for signatures, ML-KEM-768 for encryption, BLAKE3 for hashing. No ECDSA, no RSA, no curves vulnerable to Shor&apos;s algorithm. Cryptographic agility built in for future upgrades.</p>
            </div>
          </div>
        </section>

        {/* ═════════════════════════════════════════════════════ */}
        {/* COMPARISON TABLE                                      */}
        {/* ═════════════════════════════════════════════════════ */}
        <section className="comparison py-12 max-w-5xl mx-auto px-4">
          <h2 className="text-3xl md:text-4xl font-bold mb-2 text-center">How NERV Compares</h2>
          <p className="text-center text-sm opacity-60 mb-10">No compromises. Every pillar, by default.</p>

          <div className="comparison-table-wrapper">
            <table className="comparison-table w-full">
              <thead>
                <tr>
                  <th className="comparison-th comparison-th-first">Blockchain</th>
                  <th className="comparison-th">Privacy</th>
                  <th className="comparison-th">Scalability</th>
                  <th className="comparison-th">Post-Quantum</th>
                  <th className="comparison-th">Self-Improving</th>
                </tr>
              </thead>
              <tbody>
                {/* Solana / Sui */}
                <tr className="comparison-row">
                  <td className="comparison-td comparison-td-name">
                    <span className="comparison-chain-label">Solana / Sui</span>
                  </td>
                  <td className="comparison-td">
                    <span className="comparison-badge comparison-badge-no">✗</span>
                    <span className="comparison-cell-no">Transparent</span>
                  </td>
                  <td className="comparison-td">
                    <span className="comparison-badge comparison-badge-yes">✓</span>
                    <span className="comparison-cell-yes">High TPS</span>
                  </td>
                  <td className="comparison-td">
                    <span className="comparison-badge comparison-badge-no">✗</span>
                    <span className="comparison-cell-no">No</span>
                  </td>
                  <td className="comparison-td">
                    <span className="comparison-badge comparison-badge-no">✗</span>
                    <span className="comparison-cell-no">No</span>
                  </td>
                </tr>

                {/* Monero / Zcash */}
                <tr className="comparison-row">
                  <td className="comparison-td comparison-td-name">
                    <span className="comparison-chain-label">Monero / Zcash</span>
                  </td>
                  <td className="comparison-td">
                    <span className="comparison-badge comparison-badge-yes">✓</span>
                    <span className="comparison-cell-yes">Private</span>
                  </td>
                  <td className="comparison-td">
                    <span className="comparison-badge comparison-badge-no">✗</span>
                    <span className="comparison-cell-no">Low TPS</span>
                  </td>
                  <td className="comparison-td">
                    <span className="comparison-badge comparison-badge-no">✗</span>
                    <span className="comparison-cell-no">No</span>
                  </td>
                  <td className="comparison-td">
                    <span className="comparison-badge comparison-badge-no">✗</span>
                    <span className="comparison-cell-no">No</span>
                  </td>
                </tr>

                {/* NERV — highlighted */}
                <tr className="comparison-row comparison-row-nerv">
                  <td className="comparison-td comparison-td-name">
                    <span className="comparison-chain-nerv">NERV</span>
                  </td>
                  <td className="comparison-td comparison-td-nerv">
                    <span className="comparison-badge comparison-badge-yes">✓</span>
                    <span className="comparison-cell-yes">Private</span>
                  </td>
                  <td className="comparison-td comparison-td-nerv">
                    <span className="comparison-badge comparison-badge-yes">✓</span>
                    <span className="comparison-cell-yes">High TPS</span>
                  </td>
                  <td className="comparison-td comparison-td-nerv">
                    <span className="comparison-badge comparison-badge-yes">✓</span>
                    <span className="comparison-cell-yes">Yes</span>
                  </td>
                  <td className="comparison-td comparison-td-nerv">
                    <span className="comparison-badge comparison-badge-yes">✓</span>
                    <span className="comparison-cell-yes">Yes</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* ═══════════════════════════════════════════ */}
        {/* HIGH-LEVEL ARCHITECTURE — AFTER TABLE        */}
        {/* ═══════════════════════════════════════════ */}
        <section className="architecture max-w-6xl mx-auto text-center px-4">
          <h2 className="text-4xl font-bold mb-8">High-Level Architecture</h2>
          <div className="w-full rounded-xl overflow-hidden shadow-lg">
            <Image
              src="https://cdn.prod.website-files.com/64c231f464b91d6bd0303294/6711029566dc1475c0a37d98_66f258e47f53e2e2341aaae0_66d16bf1edcb81f15215c5b6_66d16b305dedb7e05c1b0920_diagram-export-8-30-2024-12_18_02-PM.png"
              alt="NERV Blockchain Architecture"
              width={1200}
              height={600}
              className="w-full h-auto"
              unoptimized
            />
          </div>
          <p className="mt-4 text-sm opacity-70">
            User → 5-hop Sphinx Mixnet → Dynamic Neural Shards → AI-Native Consensus → 512-byte Embedding Root
          </p>
        </section>

        {/* TIMELINE SECTION */}
        <section className="timeline py-12 text-center max-w-4xl mx-auto px-4">
          <h2 className="text-4xl font-bold mb-8">Road to Mainnet (100% transparent)</h2>
          <div className="timeline-items max-w-2xl mx-auto text-lg space-y-4">
            <div><span className="font-bold">Dec 2025</span> Whitepaper + all code public</div>
            <div><span className="font-bold">Q4 2026</span> Sphinx 5-hop mixnet testnet</div>
            <div><span className="font-bold">Q1 2027</span> Aurora public testnet (real metrics published)</div>
            <div><span className="font-bold">October 2027</span> Fair mainnet launch – zero pre-mine</div>
          </div>
        </section>

        {/* LINKS SECTION */}
        <section className="links py-12 text-center max-w-4xl mx-auto px-4">
          <h2 className="text-4xl font-bold mb-8">Join the nervous system</h2>
          <div className="link-grid max-w-4xl mx-auto grid md:grid-cols-2 gap-4 text-lg">
            <a href="https://github.com/nerv-bit" target="_blank" className="hover:text-cyan-400 transition p-3">GitHub Organization (10+ repos)</a>
            <a href="https://github.com/nerv-bit/formal" target="_blank" className="hover:text-cyan-400 transition p-3">Lean 4 Formal Proofs (live)</a>
            <a href="https://github.com/nerv-bit/circuits" target="_blank" className="hover:text-cyan-400 transition p-3">Post-Quantum Crypto</a>
            <a href="https://github.com/nerv-bit/simulations" target="_blank" className="hover:text-cyan-400 transition p-3">10,000-node Simulator</a>
            <a href="mailto:namsjeev@gmail.com" className="hover:text-cyan-400 transition p-3">Contact → Nerv@myself.com</a>
          </div>
        </section>

        {/* FOOTER */}
        <footer className="py-8 text-center text-sm opacity-70 border-t border-gray-800 mt-8 max-w-4xl mx-auto px-4">
          <p>© 2025–2028 NERV • All specifications, code, and proofs are MIT/Apache 2.0 or public domain</p>
          <p>No tokens exist yet • No private sales • No foundation treasury</p>
        </footer>
      </div>
    </div>
  );
}
