"use client";

import dynamic from 'next/dynamic';
import { motion } from 'framer-motion';
import Image from 'next/image';
import { useState, useEffect } from 'react';

const ParticlesComponent = dynamic(() => import('./ParticlesComponent'), {
  ssr: false,
});

export default function Home() {
  const [imageError, setImageError] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    // Check if mobile for responsive adjustments
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

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

  const neuralSymbols = ['🧠', '⚡', '🔗', '🔐', '🔒', '🔑', '💰', '📊'];

  return (
    <>
      <div className="relative min-h-screen bg-black text-white overflow-hidden">
        {/* Reduced height particle canvas - doesn't push content down */}
        <div className="absolute inset-0 z-0" style={{ height: '100vh' }}>
          <ParticlesComponent />
        </div>

        {/* Floating symbols - positioned behind content */}
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
                zIndex: 0
              }}
            >
              {symbol}
            </motion.div>
          ))}
        </div>

        {/* MAIN CONTENT - Positioned at top */}
        <div className="relative z-10">
          
          {/* HERO SECTION - Fixed position at top */}
          <section className="min-h-screen flex flex-col justify-center px-4 md:px-8 pt-20 pb-10 text-center">
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
                  className="text-7xl md:text-9xl font-bold mb-8 relative"
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
                    className="h-1 bg-gradient-to-r from-transparent via-cyan-500 to-transparent mx-auto mt-4"
                    initial={{ width: 0 }}
                    animate={{ width: "100%" }}
                    transition={{ duration: 1.5, delay: 0.5 }}
                  />
                </motion.h1>
                
                {/* Neural connection dots */}
                <div className="absolute -top-4 -left-4 w-6 h-6 bg-purple-500 rounded-full opacity-70">
                  <motion.div
                    className="absolute inset-0 bg-purple-500 rounded-full"
                    animate={{ scale: [1, 1.8, 1] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  />
                </div>
                <div className="absolute -top-4 -right-4 w-6 h-6 bg-cyan-500 rounded-full opacity-70">
                  <motion.div
                    className="absolute inset-0 bg-cyan-500 rounded-full"
                    animate={{ scale: [1, 1.8, 1] }}
                    transition={{ duration: 1.5, repeat: Infinity, delay: 0.5 }}
                  />
                </div>
              </motion.div>

              <motion.p 
                variants={childVariants} 
                className="tagline text-3xl md:text-4xl mb-6 opacity-90"
              >
                The <span className="text-cyan-400 font-semibold">living, breathing</span> financial nervous system
              </motion.p>
              
              <motion.p 
                variants={childVariants} 
                className="text-xl md:text-2xl mb-4 opacity-80"
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
                  className="btn primary bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-400 hover:to-purple-400 text-black font-semibold py-4 px-10 rounded-lg transition-all transform hover:scale-105 text-lg"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Read Whitepaper v1.01
                </motion.a>
                <motion.a
                  href="https://github.com/nerv-bit/nerv"
                  target="_blank"
                  className="btn secondary border-2 border-cyan-500 text-cyan-400 font-semibold py-4 px-10 rounded-lg transition-all transform hover:scale-105 hover:bg-cyan-500 hover:bg-opacity-10 text-lg"
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
                className="launch max-w-3xl mx-auto text-lg leading-relaxed opacity-90 mt-8 p-8 rounded-2xl border-2 border-cyan-500 border-opacity-30 bg-gradient-to-b from-black to-cyan-900 bg-opacity-20"
              >
                <span className="text-cyan-400 font-semibold">NERV</span> delivers full privacy by default (&gt;1M TPS via dynamic neural sharding and verifiable embeddings in Halo2/Nova), post-quantum security from genesis, and a self-improving useful-work economy. Fair launch June 2028: zero pre-mine, fully open-source, community-governed. Join us in building the <span className="text-purple-400 font-semibold">nervous system of the private internet</span>!
              </motion.p>
            </motion.div>
          </section>

          {/* ARCHITECTURE SECTION - With fallback image handling */}
          <motion.section
            variants={sectionVariants}
            initial="offscreen"
            whileInView="onscreen"
            viewport={{ once: true, amount: 0.3 }}
            className="architecture py-20 px-4 md:px-8 text-center"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className="max-w-6xl mx-auto"
            >
              <h2 className="text-4xl md:text-5xl font-bold mb-12">
                <span className="bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                  High-Level Architecture
                </span>
              </h2>
              
              <motion.div 
                className="w-full relative rounded-2xl overflow-hidden border-2 border-cyan-500 border-opacity-30 bg-gradient-to-br from-black to-cyan-900 bg-opacity-20 p-4 md:p-8"
                whileHover={{ scale: 1.005 }}
              >
                {/* Large container for diagram */}
                <div className="relative w-full min-h-[500px] md:min-h-[700px] bg-black bg-opacity-50 rounded-xl flex items-center justify-center">
                  {imageError ? (
                    // Fallback if image fails to load
                    <div className="text-center p-8">
                      <div className="text-6xl mb-4">🔄</div>
                      <h3 className="text-2xl text-cyan-400 mb-2">Architecture Diagram</h3>
                      <p className="text-gray-400 mb-4">User → 5-hop TEE Mixer → Dynamic Neural Shards → AI-Native Consensus → 512-byte Embedding Root</p>
                      <a 
                        href="https://cdn.prod.website-files.com/64c231f464b91d6bd0303294/6711029566dc1475c0a37d98_66f258e47f53e2e2341aaae0_66d16bf1edcb81f15215c5b6_66d16b305dedb7e05c1b0920_diagram-export-8-30-2024-12_18_02-PM.png" 
                        target="_blank" 
                        className="text-cyan-400 hover:text-cyan-300 underline"
                      >
                        View diagram in new tab
                      </a>
                    </div>
                  ) : (
                    // Try loading with Image component first
                    <>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <Image 
                          src="https://cdn.prod.website-files.com/64c231f464b91d6bd0303294/6711029566dc1475c0a37d98_66f258e47f53e2e2341aaae0_66d16bf1edcb81f15215c5b6_66d16b305dedb7e05c1b0920_diagram-export-8-30-2024-12_18_02-PM.png"
                          alt="NERV Blockchain Architecture Diagram"
                          fill
                          style={{ 
                            objectFit: 'contain',
                            objectPosition: 'center',
                            padding: isMobile ? '10px' : '30px'
                          }}
                          sizes="(max-width: 768px) 100vw, 1200px"
                          priority
                          className="rounded-lg"
                          unoptimized={true}
                          onError={() => setImageError(true)}
                        />
                      </div>
                      {/* Fallback img tag hidden by default */}
                      <img 
                        src="https://cdn.prod.website-files.com/64c231f464b91d6bd0303294/6711029566dc1475c0a37d98_66f258e47f53e2e2341aaae0_66d16bf1edcb81f15215c5b6_66d16b305dedb7e05c1b0920_diagram-export-8-30-2024-12_18_02-PM.png" 
                        alt="NERV Architecture Diagram Fallback"
                        className="absolute inset-0 w-full h-full object-contain opacity-0"
                        onLoad={(e) => {
                          // If this loads but Image fails, show it
                          if (imageError) {
                            e.currentTarget.classList.remove('opacity-0');
                          }
                        }}
                      />
                    </>
                  )}
                </div>
                
                <motion.div 
                  className="mt-8 text-base md:text-lg opacity-90 border-t border-cyan-500 border-opacity-20 pt-8"
                  initial={{ opacity: 0 }}
                  whileInView={{ opacity: 1 }}
                  transition={{ delay: 0.3 }}
                >
                  <div className="grid grid-cols-1 md:grid-cols-5 gap-4 items-center justify-items-center">
                    <span className="text-cyan-400 font-bold text-lg">User</span>
                    <div className="text-gray-400">→</div>
                    <span className="text-purple-400 font-bold text-lg">5-hop TEE Mixer</span>
                    <div className="text-gray-400">→</div>
                    <span className="text-green-400 font-bold text-lg">Dynamic Neural Shards</span>
                    <div className="text-gray-400">→</div>
                    <span className="text-yellow-400 font-bold text-lg">AI-Native Consensus</span>
                    <div className="text-gray-400">→</div>
                    <span className="text-cyan-400 font-bold text-lg">512-byte Embedding Root</span>
                  </div>
                </motion.div>
              </motion.div>
            </motion.div>
          </motion.section>

          {/* Rest of your sections remain the same... */}
          {/* ... [Keep all other sections unchanged] ... */}
        </div>
      </div>
    </>
  );
}
