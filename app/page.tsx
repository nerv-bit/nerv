"use client";

import { useState } from 'react';
import Image from 'next/image';
import NeuralBackground from './NeuralBackground';

export default function Home() {
  const [heroHovered, setHeroHovered] = useState(false);

  return (
    <div className="relative min-h-screen bg-black text-white overflow-x-hidden">
      {/* Neural Background - Absolutely positioned */}
      <div className="absolute inset-0 z-0">
        <NeuralBackground isActive={heroHovered} />
      </div>

      {/* HERO SECTION - FORCED TO ABSOLUTE TOP */}
      <section 
        className="absolute -top-8 left-1/2 transform -translate-x-1/2 w-full max-w-5xl px-4 text-center"
        onMouseEnter={() => setHeroHovered(true)}
        onMouseLeave={() => setHeroHovered(false)}
      >
        {/* Cryptographic Icons */}
        <div className="crypto-icon absolute left-[5%] top-1/4 text-3xl opacity-0">🔐</div>
        <div className="crypto-icon absolute right-[7%] top-1/3 text-2xl opacity-0">{`</>`}</div>
        <div className="crypto-icon absolute left-[10%] bottom-1/3 text-xl opacity-0">{`{ }`}</div>
        <div className="crypto-icon absolute right-[10%] bottom-1/4 text-3xl opacity-0">🛡️</div>

        {/* Glass Hero Block - Minimal padding */}
        <div className="glass-hero mt-0 pt-4 pb-8">
          <h1 className="text-7xl md:text-9xl font-bold mb-2 tracking-tighter">
            NERV
          </h1>
          <div className="h-1 w-24 bg-gradient-to-r from-cyan-500 to-purple-500 mx-auto my-6 rounded-full"></div>
          
          {/* Original Hero Text */}
          <p className="tagline text-2xl md:text-3xl mb-4 opacity-90">
            The private, post-quantum, infinitely scalable blockchain
          </p>
          <p className="launch text-lg md:text-xl mb-12 opacity-80">
            Fair launch June 2028 • No pre-mine • Epics, user-stories, tasks public today
          </p>

          {/* Original Buttons */}
          <div className="buttons flex flex-col sm:flex-row gap-6 justify-center mb-8">
            <a
              href="https://github.com/nerv-bit/nerv/blob/main/NERV_Whitepaper_v1.01.pdf"
              className="btn primary bg-cyan-500 hover:bg-cyan-400 text-black font-semibold py-3 px-8 rounded-lg transition hover:scale-105"
            >
              Read Whitepaper v1.01
            </a>
            <a
              href="https://github.com/nerv-bit/nerv"
              target="_blank"
              className="btn secondary border border-purple-500 text-white font-semibold py-3 px-8 rounded-lg transition hover:scale-105"
            >
              GitHub →
            </a>
          </div>

          <p className="mt-8 text-lg leading-relaxed opacity-90 max-w-3xl mx-auto">
            <span className="gradient-text font-semibold">NERV</span> delivers full privacy by default (&gt;1M TPS via dynamic neural sharding and verifiable embeddings in Halo2/Nova), post-quantum security from genesis, and a self-improving useful-work economy. Fair launch June 2028: zero pre-mine, fully open-source, community-governed. Join us in building the <span className="gradient-text font-semibold">nervous system of the private internet</span>!
          </p>
        </div>
      </section>

      {/* MAIN CONTENT - Pushed down to avoid hero overlap */}
      <div className="relative z-10 pt-[480px] md:pt-[520px]">
        
        {/* HIGH-LEVEL ARCHITECTURE SECTION */}
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
            User → 5-hop TEE Mixer → Dynamic Neural Shards → AI-Native Consensus → 512-byte Embedding Root
          </p>
        </section>

        {/* CORE INNOVATIONS SECTION */}
        <section className="promise py-12 text-center max-w-5xl mx-auto px-4">
          <h2 className="text-4xl font-bold mb-8">Core Pillars of NERV</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="neural-card p-6">
              <div className="text-4xl mb-4">🧠</div>
              <h3 className="text-xl font-bold mb-3 gradient-text">Live, Breathing System</h3>
              <p className="opacity-80">A self-adapting, economic nervous system that evolves organically.</p>
            </div>
            <div className="neural-card p-6">
              <div className="text-4xl mb-4">⚡</div>
              <h3 className="text-xl font-bold mb-3 gradient-text">Neural Embeddings</h3>
              <p className="opacity-80">State compression and verification through AI-native transformer models.</p>
            </div>
            <div className="neural-card p-6">
              <div className="text-4xl mb-4">🔐</div>
              <h3 className="text-xl font-bold mb-3 gradient-text">Cryptography</h3>
              <p className="opacity-80">Post-quantum, zero-trust security layered from the genesis block.</p>
            </div>
          </div>
        </section>

        {/* TIMELINE SECTION */}
        <section className="timeline py-12 text-center max-w-4xl mx-auto px-4">
          <h2 className="text-4xl font-bold mb-8">Road to Mainnet (100% transparent)</h2>
          <div className="timeline-items max-w-2xl mx-auto text-lg space-y-4">
            <div><span className="font-bold">Dec 2025</span> Whitepaper + all code & proofs public</div>
            <div><span className="font-bold">Q1 2026</span> First multi-vendor TEE mixer testnet</div>
            <div><span className="font-bold">Q2 2026</span> Aurora public testnet (real metrics published)</div>
            <div><span className="font-bold">June 2028</span> Fair mainnet launch – zero pre-mine</div>
          </div>
        </section>

        {/* LINKS SECTION */}
        <section className="links py-12 text-center max-w-4xl mx-auto px-4">
          <h2 className="text-4xl font-bold mb-8">Join the nervous system</h2>
          <div className="link-grid max-w-4xl mx-auto grid md:grid-cols-2 gap-4 text-lg">
            <a href="https://github.com/nerv-bit" target="_blank" className="hover:text-cyan-400 transition p-3">GitHub Organization (10+ repos)</a>
            <a href="https://github.com/nerv-bit/formal" target="_blank" className="hover:text-cyan-400 transition p-3">Lean 4 Formal Proofs (live)</a>
            <a href="https://github.com/nerv-bit/circuits" target="_blank" className="hover:text-cyan-400 transition p-3">Halo2 Circuits</a>
            <a href="https://github.com/nerv-bit/simulations" target="_blank" className="hover:text-cyan-400 transition p-3">10 000-node Simulator</a>
            <a href="mailto:namsjeev@gmail.com" className="hover:text-cyan-400 transition p-3">Contact → namsjeev@gmail.com</a>
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
