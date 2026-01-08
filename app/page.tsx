import dynamic from 'next/dynamic';
import { animated, useSpring, config } from '@react-spring/web';
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
    visible: { opacity: 1, y: 0, transition: { duration: 0.8 } },
  };

  const sectionVariants = {
    offscreen: { opacity: 0, y: 50 },
    onscreen: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.8, ease: 'easeOut' },
    },
  };

  return (
    <>
      <div className="relative min-h-screen bg-black text-white overflow-hidden">
        <ParticlesComponent />

        <div className="hero relative z-10 py-20 px-8 text-center">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="max-w-5xl mx-auto"
          >
            <motion.h1
              variants={childVariants}
              className="text-6xl md:text-8xl font-bold glow-pulse mb-6"
            >
              NERV
            </motion.h1>
            <motion.p variants={childVariants} className="tagline text-2xl md:text-3xl mb-4 opacity-90">
              The private, post-quantum, infinitely scalable blockchain
            </motion.p>
            <motion.p variants={childVariants} className="launch text-lg md:text-xl mb-12 opacity-80">
              Fair launch June 2028 • No pre-mine • Epics, user-stories, tasks public today
            </motion.p>

            <motion.div variants={childVariants} className="buttons flex flex-col sm:flex-row gap-6 justify-center mb-16">
              <motion.a
                href="https://github.com/nerv-bit/nerv/blob/main/NERV_Whitepaper_v1.01.pdf"
                className="btn primary bg-cyan-500 hover:bg-cyan-400 text-black font-semibold py-4 px-8 rounded-lg transition btn-primary"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Read Whitepaper v1.01
              </motion.a>
              <motion.a
                href="https://github.com/nerv-bit/nerv"
                target="_blank"
                className="btn secondary border border-purple-500 text-white font-semibold py-4 px-8 rounded-lg transition btn-secondary"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                GitHub →
              </motion.a>
            </motion.div>

            <div className="image-section flex flex-col md:flex-row gap-8 justify-center items-center mt-12">
              <motion.div 
                variants={childVariants}
                className="w-full md:w-1/2 h-64 md:h-80 relative"
              >
                <Image 
                  src="/hero.svg" 
                  alt="Hero" 
                  fill
                  style={{ objectFit: 'contain' }}
                  sizes="(max-width: 768px) 100vw, 50vw"
                  priority
                />
              </motion.div>
              <motion.div 
                variants={childVariants}
                className="w-full md:w-1/2 h-64 md:h-80 relative"
              >
                <Image 
                  src="/hero2.svg" 
                  alt="Hero 2" 
                  fill
                  style={{ objectFit: 'contain' }}
                  sizes="(max-width: 768px) 100vw, 50vw"
                  priority
                />
              </motion.div>
            </div>

            <motion.p
              variants={childVariants}
              className="launch max-w-3xl mx-auto text-lg leading-relaxed opacity-90 mt-12"
            >
              NERV delivers full privacy by default (&gt;1M TPS via dynamic neural sharding and verifiable embeddings in Halo2/Nova), post-quantum security from genesis, and a self-improving useful-work economy. Fair launch June 2028: zero pre-mine, fully open-source, community-governed. Join us in building the nervous system of the private internet!
            </motion.p>
          </motion.div>

          <motion.section
            variants={sectionVariants}
            initial="offscreen"
            whileInView="onscreen"
            viewport={{ once: true, amount: 0.3 }}
            className="architecture mt-32 max-w-5xl mx-auto text-center"
          >
            <h2 className="text-4xl font-bold mb-8 glow-pulse">High-Level Architecture</h2>
            <motion.div 
              className="w-full relative h-96 rounded-xl overflow-hidden shadow-pulse"
              whileHover={{ scale: 1.02 }}
            >
              <Image 
                src="https://cdn.prod.website-files.com/64c231f464b91d6bd0303294/6711029566dc1475c0a37d98_66f258e47f53e2e2341aaae0_66d16bf1edcb81f15215c5b6_66d16b305dedb7e05c1b0920_diagram-export-8-30-2024-12_18_02-PM.png"
                alt="NERV Blockchain Architecture"
                fill
                style={{ objectFit: 'cover' }}
                sizes="100vw"
              />
            </motion.div>
            <p className="mt-4 text-sm opacity-70">
              User → 5-hop TEE Mixer → Dynamic Neural Shards → AI-Native Consensus → 512-byte Embedding Root
            </p>
          </motion.section>
        </div>

        <motion.section 
          variants={sectionVariants} 
          initial="offscreen" 
          whileInView="onscreen" 
          viewport={{ once: true }} 
          className="promise py-20 text-center"
        >
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto text-xl font-medium">
            <motion.div variants={childVariants}>Privacy by default</motion.div>
            <motion.div variants={childVariants}>1 M+ TPS target</motion.div>
            <motion.div variants={childVariants}>Useful-work economy</motion.div>
            <motion.div variants={childVariants}>Quantum-secure from day 0</motion.div>
          </div>
        </motion.section>

        <motion.section 
          variants={sectionVariants} 
          initial="offscreen" 
          whileInView="onscreen" 
          viewport={{ once: true }} 
          className="timeline py-20 text-center"
        >
          <h2 className="text-4xl font-bold mb-12 glow-pulse">Road to Mainnet (100% transparent)</h2>
          <div className="timeline-items max-w-2xl mx-auto text-lg space-y-6">
            <motion.div variants={childVariants}><span className="font-bold">Dec 2025</span> Whitepaper + all code & proofs public</motion.div>
            <motion.div variants={childVariants}><span className="font-bold">Q1 2026</span> First multi-vendor TEE mixer testnet</motion.div>
            <motion.div variants={childVariants}><span className="font-bold">Q2 2026</span> Aurora public testnet (real metrics published)</motion.div>
            <motion.div variants={childVariants}><span className="font-bold">June 2028</span> Fair mainnet launch – zero pre-mine</motion.div>
          </div>
        </motion.section>

        <motion.section 
          variants={sectionVariants} 
          initial="offscreen" 
          whileInView="onscreen" 
          viewport={{ once: true }} 
          className="links py-20 text-center"
        >
          <h2 className="text-4xl font-bold mb-12 glow-pulse">Join the nervous system</h2>
          <div className="link-grid max-w-4xl mx-auto grid md:grid-cols-2 gap-6 text-lg">
            <motion.a href="https://github.com/nerv-bit" target="_blank" className="hover:text-cyan-400 transition" variants={childVariants}>GitHub Organization (10+ repos)</motion.a>
            <motion.a href="https://github.com/nerv-bit/formal" target="_blank" className="hover:text-cyan-400 transition" variants={childVariants}>Lean 4 Formal Proofs (live)</motion.a>
            <motion.a href="https://github.com/nerv-bit/circuits" target="_blank" className="hover:text-cyan-400 transition" variants={childVariants}>Halo2 Circuits</motion.a>
            <motion.a href="https://github.com/nerv-bit/simulations" target="_blank" className="hover:text-cyan-400 transition" variants={childVariants}>10 000-node Simulator</motion.a>
            <motion.a href="mailto:namsjeev@gmail.com" className="hover:text-cyan-400 transition" variants={childVariants}>Contact → namsjeev@gmail.com</motion.a>
          </div>
        </motion.section>

        <footer className="py-12 text-center text-sm opacity-70">
          <p>© 2025–2028 NERV • All specifications, code, and proofs are MIT/Apache 2.0 or public domain</p>
          <p>No tokens exist yet • No private sales • No foundation treasury</p>
        </footer>
      </div>
    </>
  );
}
