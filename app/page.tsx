"use client";

import { useState } from 'react';
import Image from 'next/image';
import NeuralBackground from './NeuralBackground';

export default function Home() {
  const [heroHovered, setHeroHovered] = useState(false);

  return (
    <div className="relative min-h-screen bg-black text-white overflow-x-hidden">
      {/* 1. Interactive Neural Background */}
      <NeuralBackground isActive={heroHovered} />

      {/* 2. Main Content Container */}
      <div className="relative z-10 container mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* HERO SECTION - With hover-triggered animation */}
        <section 
          className="min-h-[80vh] flex flex-col justify-center items-center text-center"
          onMouseEnter={() => setHeroHovered(true)}
          onMouseLeave={() => setHeroHovered(false)}
        >
          {/* Cryptographic Icons (appear on hover) */}
          <div className="crypto-icon absolute left-[10%] top-1/4 text-3xl opacity-0">🔐</div>
          <div className="crypto-icon absolute right-[12%] top-1/3 text-2xl opacity-0">{`</>`}</div>
          <div className="crypto-icon absolute left-[15%] bottom-1/3 text-xl opacity-0">{`{ }`}</div>
          <div className="crypto-icon absolute right-[15%] bottom-1/4 text-3xl opacity-0">🛡️</div>

          {/* Glass Hero Block */}
          <div className="glass-hero">
            <h1 className="text-7xl md:text-9xl font-bold mb-2 tracking-tighter">
              NERV
            </h1>
            <div className="h-1 w-24 bg-gradient-to-r from-cyan-500 to-purple-500 mx-auto my-6 rounded-full"></div>
            
            {/* YOUR ORIGINAL HERO TEXT - UNCHANGED */}
            <p className="tagline text-2xl md:text-3xl mb-4 opacity-90">
              The private, post-quantum, infinitely scalable blockchain
            </p>
            <p className="launch text-lg md:text-xl mb-12 opacity-80">
              Fair launch June 2028 • No pre-mine • Epics, user-stories, tasks public today
            </p>

            {/* YOUR ORIGINAL BUTTONS - UNCHANGED */}
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

        {/* HIGH-LEVEL ARCHITECTURE SECTION - YOUR ORIGINAL, UNCHANGED */}
        <section className="architecture mt-20 max-w-6xl mx-auto text-center">
          <h2 className="text-4xl font-bold mb-8">High-Level Architecture</h2>
          {/* Your original diagram container and image tag remain exactly as they were */}
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

        {/* CORE INNOVATIONS SECTION - Restyled Cards */}
        <section className="promise py-20 text-center max-w-5xl mx-auto">
          <h2 className="text-4xl font-bold mb-12">Core Pillars of NERV</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Card 1: Living System */}
            <div className="neural-card p-6">
              <div className="text-4xl mb-4">🧠</div>
              <h3 className="text-xl font-bold mb-3 gradient-text">Live, Breathing System</h3>
              <p className="opacity-80">A self-adapting, economic nervous system that evolves organically.</p>
            </div>
            {/* Card 2: Neural Embeddings */}
            <div className="neural-card p-6">
              <div className="text-4xl mb-4">⚡</div>
              <h3 className="text-xl font-bold mb-3 gradient-text">Neural Embeddings</h3>
              <p className="opacity-80">State compression and verification through AI-native transformer models.</p>
            </div>
            {/* Card 3: Cryptography */}
            <div className="neural-card p-6">
              <div className="text-4xl mb-4">🔐</div>
              <h3 className="text-xl font-bold mb-3 gradient-text">Cryptography</h3>
              <p className="opacity-80">Post-quantum, zero-trust security layered from the genesis block.</p>
            </div>
          </div>
        </section>

        {/* YOUR ORIGINAL TIMELINE, LINKS, AND FOOTER SECTIONS */}
        {/* ... Paste the rest of your original page.tsx sections here exactly as they were ... */}
        {/* They will automatically inherit the improved spacing and container styles */}

      </div>
    </div>
  );
}
