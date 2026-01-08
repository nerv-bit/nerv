"use client";

import dynamic from 'next/dynamic';
import { motion } from 'framer-motion';
import Image from 'next/image';

const ParticlesComponent = dynamic(() => import('./ParticlesComponent'), {
  ssr: false,
});

export default function Home() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.3,
      },
    },
  };

  const childVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: { 
      opacity: 1, 
      y: 0, 
      transition: { 
        duration: 0.8,
        ease: "easeOut"
      } 
    },
  };

  const sectionVariants = {
    offscreen: { opacity: 0, y: 50 },
    onscreen: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.8, ease: 'easeOut' },
    },
  };

  // Neural/crypto symbols for floating animation
  const neuralSymbols = ['🧠', '⚡', '🔗', '🔐', '🔒', '🔑', '💰', '📊'];

  return (
    <>
      <div className="relative min-h-screen bg-black text-white overflow-hidden">
        <ParticlesComponent />

        {/* Floating neural/crypto symbols */}
        <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none">
          {neuralSymbols.map((symbol, index) => (
            <motion.div
              key={index}
              className="absolute text-2xl md:text-3xl opacity-10"
              initial={{ 
                x: Math.random() * 100 + '%', 
                y: Math.random() * 100 + '%',
                rotate: 0 
              }}
              animate={{ 
                y: [null, '-20px', '0px'],
                rotate: 360,
              }}
              transition={{
                y: {
                  duration: 10 + index * 2,
                  repeat: Infinity,
                  repeatType: "reverse",
                  ease: "easeInOut"
                },
                rotate: {
                  duration: 20 + index * 5,
                  repeat: Infinity,
                  ease: "linear"
                }
              }}
              style={{
                animationDelay: `${index * 2}s`,
              }}
            >
              {symbol}
            </motion.div>
          ))}
        </div>

        <div className="hero relative z-10 py-20 px-8 text-center">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="max-w-5xl mx-auto"
          >
            {/* Animated title with neural pulse */}
            <motion.div
              variants={childVariants}
              className="relative inline-block"
            >
              <motion.h1
                className="text-6xl md:text-8xl font-bold mb-6 relative"
                animate={{
                  textShadow: [
                    "0 0 10px rgba(0, 255, 255, 0.5)",
                    "0 0 20px rgba(0, 255, 255, 0.8)",
                    "0 0 10px rgba(0, 255, 255, 0.5)"
                  ]
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  repeatType: "reverse"
                }}
              >
                NERV
                {/* Animated underline for "living system" */}
                <motion.div 
                  className="h-1 bg-gradient-to-r from-transparent via-cyan-500 to-transparent mx-auto mt-2"
                  initial={{ width: 0 }}
                  animate={{ width: "100%" }}
                  transition={{ duration: 1.5, delay: 0.5 }}
                />
              </motion.h1>
              
              {/* Neural connection dots */}
              <div className="absolute -top-2 -left-2 w-4 h-4 bg-purple-500 rounded-full opacity-70">
                <motion.div
                  className="absolute inset-0 bg-purple-500 rounded-full"
                  animate={{ scale: [1, 1.5, 1] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                />
              </div>
              <div className="absolute -top-2 -right-2 w-4 h-4 bg-cyan-500 rounded-full opacity-70">
                <motion.div
                  className="absolute inset-0 bg-cyan-500 rounded-full"
                  animate={{ scale: [1, 1.5, 1] }}
                  transition={{ duration: 1.5, repeat: Infinity, delay: 0.5 }}
                />
              </div>
            </motion.div>

            <motion.p 
              variants={childVariants} 
              className="tagline text-2xl md:text-3xl mb-4 opacity-90"
            >
              The <span className="text-cyan-400 font-semibold">living, breathing</span> financial nervous system
            </motion.p>
            
            <motion.p 
              variants={childVariants} 
              className="text-xl md:text-2xl mb-2 opacity-80"
            >
              Powered by <span className="text-purple-400 font-semibold">neural transformer embeddings</span>
            </motion.p>
            
            <motion.p 
              variants={childVariants} 
              className="text-lg md:text-xl mb-12 opacity-70"
            >
              Secured by <span className="text-green-400 font-semibold">post-quantum cryptography</span>
            </motion.p>

            <motion.p variants={childVariants} className="launch text-lg md:text-xl mb-12 opacity-80">
              Fair launch June 2028 • No pre-mine • Epics, user-stories, tasks public today
            </motion.p>

            <motion.div variants={childVariants} className="buttons flex flex-col sm:flex-row gap-6 justify-center mb-16">
              <motion.a
                href="https://github.com/nerv-bit/nerv/blob/main/NERV_Whitepaper_v1.01.pdf"
                className="btn primary bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-400 hover:to-purple-400 text-black font-semibold py-4 px-8 rounded-lg transition-all transform hover:scale-105"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Read Whitepaper v1.01
              </motion.a>
              <motion.a
                href="https://github.com/nerv-bit/nerv"
                target="_blank"
                className="btn secondary border border-cyan-500 text-cyan-400 font-semibold py-4 px-8 rounded-lg transition-all transform hover:scale-105 hover:bg-cyan-500 hover:bg-opacity-10"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <span className="flex items-center gap-2">
                  Explore the Code 
                  <motion.span
                    animate={{ x: [0, 5, 0] }}
                    transition={{ duration: 1, repeat: Infinity }}
                  >
                    →
                  </motion.span>
                </span>
              </motion.a>
            </motion.div>

            <motion.p
              variants={childVariants}
              className="launch max-w-3xl mx-auto text-lg leading-relaxed opacity-90 mt-12 p-6 rounded-xl border border-cyan-500 border-opacity-30 bg-gradient-to-b from-black to-cyan-900 bg-opacity-10"
            >
              <span className="text-cyan-400 font-semibold">NERV</span> delivers full privacy by default (&gt;1M TPS via dynamic neural sharding and verifiable embeddings in Halo2/Nova), post-quantum security from genesis, and a self-improving useful-work economy. Fair launch June 2028: zero pre-mine, fully open-source, community-governed. Join us in building the <span className="text-purple-400 font-semibold">nervous system of the private internet</span>!
            </motion.p>
          </motion.div>

          <motion.section
            variants={sectionVariants}
            initial="offscreen"
            whileInView="onscreen"
            viewport={{ once: true, amount: 0.3 }}
            className="architecture mt-32 max-w-5xl mx-auto text-center"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
            >
              <h2 className="text-4xl font-bold mb-8">
                <span className="bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                  High-Level Architecture
                </span>
              </h2>
              
              {/* Fixed Architecture Diagram */}
              <motion.div 
                className="w-full relative rounded-xl overflow-hidden border border-cyan-500 border-opacity-30 bg-gradient-to-br from-black to-cyan-900 bg-opacity-20 p-4"
                whileHover={{ scale: 1.01 }}
              >
                <div className="relative h-[500px] w-full">
                  <Image 
                    src="https://cdn.prod.website-files.com/64c231f464b91d6bd0303294/6711029566dc1475c0a37d98_66f258e47f53e2e2341aaae0_66d16bf1edcb81f15215c5b6_66d16b305dedb7e05c1b0920_diagram-export-8-30-2024-12_18_02-PM.png"
                    alt="NERV Blockchain Architecture Diagram showing: User → 5-hop TEE Mixer → Dynamic Neural Shards → AI-Native Consensus → 512-byte Embedding Root"
                    fill
                    style={{ objectFit: 'contain' }}
                    sizes="(max-width: 768px) 100vw, 800px"
                    priority
                    className="rounded-lg"
                  />
                </div>
                <motion.div 
                  className="mt-4 text-sm opacity-70 border-t border-cyan-500 border-opacity-20 pt-4"
                  initial={{ opacity: 0 }}
                  whileInView={{ opacity: 1 }}
                  transition={{ delay: 0.3 }}
                >
                  <span className="text-cyan-400">User</span> → <span className="text-purple-400">5-hop TEE Mixer</span> → <span className="text-green-400">Dynamic Neural Shards</span> → <span className="text-yellow-400">AI-Native Consensus</span> → <span className="text-cyan-400">512-byte Embedding Root</span>
                </motion.div>
              </motion.div>
            </motion.div>
          </motion.section>
        </div>

        <motion.section 
          variants={sectionVariants} 
          initial="offscreen" 
          whileInView="onscreen" 
          viewport={{ once: true }} 
          className="promise py-20 text-center relative"
        >
          <div className="max-w-4xl mx-auto">
            <h2 className="text-4xl font-bold mb-12">
              <span className="bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
                Core Innovations
              </span>
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-lg font-medium">
              {[
                { 
                  title: "Living Financial Nervous System", 
                  desc: "Self-healing, adaptive network topology",
                  icon: "🧠",
                  color: "from-purple-500 to-pink-500"
                },
                { 
                  title: "Neural Transformer Embeddings", 
                  desc: "AI-native state representation & compression",
                  icon: "⚡",
                  color: "from-cyan-500 to-blue-500"
                },
                { 
                  title: "Post-Quantum Cryptography", 
                  desc: "Zero-trust, quantum-resistant from genesis",
                  icon: "🔐",
                  color: "from-green-500 to-emerald-500"
                }
              ].map((item, index) => (
                <motion.div 
                  key={index}
                  variants={childVariants}
                  className="p-6 rounded-xl border border-opacity-20 bg-gradient-to-b from-black to-gray-900 bg-opacity-50"
                  whileHover={{ 
                    scale: 1.05,
                    borderColor: "rgba(0, 255, 255, 0.3)",
                    boxShadow: "0 0 20px rgba(0, 255, 255, 0.1)"
                  }}
                >
                  <div className={`text-4xl mb-4 bg-gradient-to-r ${item.color} bg-clip-text text-transparent`}>
                    {item.icon}
                  </div>
                  <h3 className="text-xl font-bold mb-2">{item.title}</h3>
                  <p className="text-sm opacity-70">{item.desc}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.section>

        {/* Rest of your sections remain the same */}
        <motion.section 
          variants={sectionVariants} 
          initial="offscreen" 
          whileInView="onscreen" 
          viewport={{ once: true }} 
          className="timeline py-20 text-center"
        >
          <h2 className="text-4xl font-bold mb-12">
            <span className="bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
              Road to Mainnet (100% transparent)
            </span>
          </h2>
          <div className="timeline-items max-w-2xl mx-auto text-lg space-y-6">
            {[
              { date: "Dec 2025", desc: "Whitepaper + all code & proofs public" },
              { date: "Q1 2026", desc: "First multi-vendor TEE mixer testnet" },
              { date: "Q2 2026", desc: "Aurora public testnet (real metrics published)" },
              { date: "June 2028", desc: "Fair mainnet launch – zero pre-mine" }
            ].map((item, index) => (
              <motion.div 
                key={index}
                variants={childVariants}
                className="flex items-center gap-4 p-4 rounded-lg border border-cyan-500 border-opacity-20 hover:border-opacity-40 transition-all"
                whileHover={{ x: 10 }}
              >
                <span className="font-bold text-cyan-400 min-w-[100px]">{item.date}</span>
                <span className="text-left">{item.desc}</span>
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* Links and Footer sections remain similar to before */}
        <motion.section 
          variants={sectionVariants} 
          initial="offscreen" 
          whileInView="onscreen" 
          viewport={{ once: true }} 
          className="links py-20 text-center"
        >
          <h2 className="text-4xl font-bold mb-12">
            <span className="bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
              Join the Nervous System
            </span>
          </h2>
          <div className="link-grid max-w-4xl mx-auto grid md:grid-cols-2 gap-6 text-lg">
            {[
              { label: "GitHub Organization (10+ repos)", href: "https://github.com/nerv-bit" },
              { label: "Lean 4 Formal Proofs (live)", href: "https://github.com/nerv-bit/formal" },
              { label: "Halo2 Circuits", href: "https://github.com/nerv-bit/circuits" },
              { label: "10,000-node Simulator", href: "https://github.com/nerv-bit/simulations" },
              { label: "Contact → namsjeev@gmail.com", href: "mailto:namsjeev@gmail.com" }
            ].map((link, index) => (
              <motion.a 
                key={index}
                href={link.href} 
                target={link.href.startsWith('http') ? '_blank' : undefined}
                className="hover:text-cyan-400 transition-all p-3 rounded-lg border border-transparent hover:border-cyan-500 hover:border-opacity-30"
                variants={childVariants}
                whileHover={{ scale: 1.02 }}
              >
                {link.label}
              </motion.a>
            ))}
          </div>
        </motion.section>

        <footer className="py-12 text-center text-sm opacity-70 border-t border-cyan-500 border-opacity-20">
          <p>© 2025–2028 NERV • All specifications, code, and proofs are MIT/Apache 2.0 or public domain</p>
          <p>No tokens exist yet • No private sales • No foundation treasury</p>
        </footer>
      </div>
    </>
  );
}
