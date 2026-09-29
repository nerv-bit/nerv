"use client";

import { useEffect, useRef } from 'react';

interface Particle {
  x: number;
  y: number;
  size: number;
  speedX: number;
  speedY: number;
  color: string;
  pulseOffset: number;
  isCrypto: boolean;
  symbol: string;
}

const ParticlesComponent = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const particlesRef = useRef<Particle[]>([]);
  const timeRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas dimensions
    const setCanvasSize = () => {
      if (canvas) {
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;
      }
    };

    setCanvasSize();
    window.addEventListener('resize', setCanvasSize);

    // Initialize particles with themes
    const initParticles = () => {
      particlesRef.current = [];
      const particleCount = Math.min(150, Math.floor((canvas.width * canvas.height) / 8000));
      
      // Symbols for cryptographic theme
      const cryptoSymbols = ['🔐', '🔒', '🔑', '💰', '⚡', '🧬', '🧠', '🔗'];
      
      for (let i = 0; i < particleCount; i++) {
        const isCrypto = Math.random() < 0.2; // 20% are cryptographic symbols
        particlesRef.current.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          size: Math.random() * 3 + 1,
          speedX: (Math.random() - 0.5) * 0.8,
          speedY: (Math.random() - 0.5) * 0.8,
          color: `hsl(${Math.random() * 60 + 180}, 100%, ${Math.random() * 30 + 60}%)`,
          pulseOffset: Math.random() * Math.PI * 2,
          isCrypto,
          symbol: cryptoSymbols[Math.floor(Math.random() * cryptoSymbols.length)],
        });
      }
    };

    // Draw neural connections (living system theme)
    const drawNeuralConnections = (ctx: CanvasRenderingContext2D, particles: Particle[], time: number) => {
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          // Neural connections with pulsing effect
          if (distance < 120) {
            const pulseIntensity = (Math.sin(time * 0.002 + particles[i].pulseOffset) + 1) * 0.5;
            const alpha = 0.2 * pulseIntensity * (1 - distance / 120);
            
            ctx.beginPath();
            ctx.strokeStyle = `rgba(100, 200, 255, ${alpha})`;
            ctx.lineWidth = 0.3 + pulseIntensity * 0.7;
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }
    };

    // Draw particles
    const drawParticles = (time: number) => {
      if (!canvas || !ctx) return;

      // Semi-transparent trail for "breathing" effect
      ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const particles = particlesRef.current;
      
      // Draw neural connections
      drawNeuralConnections(ctx, particles, time);
      
      // Update and draw particles
      particles.forEach(particle => {
        // Breathing pulse effect
        const pulse = (Math.sin(time * 0.003 + particle.pulseOffset) + 1) * 0.5;
        const currentSize = particle.size + pulse * 2;
        
        // Transformer embeddings effect - particles move in wave patterns
        particle.x += particle.speedX + Math.sin(time * 0.001 + particle.y * 0.01) * 0.3;
        particle.y += particle.speedY + Math.cos(time * 0.001 + particle.x * 0.01) * 0.3;
        
        // Boundary check with bounce
        if (particle.x < 0 || particle.x > canvas.width) {
          particle.speedX *= -1;
          particle.x = particle.x < 0 ? 0 : canvas.width;
        }
        if (particle.y < 0 || particle.y > canvas.height) {
          particle.speedY *= -1;
          particle.y = particle.y < 0 ? 0 : canvas.height;
        }
        
        if (particle.isCrypto) {
          // Draw cryptographic symbols
          ctx.font = `${currentSize * 4}px Arial`;
          ctx.fillStyle = particle.color;
          ctx.fillText(particle.symbol, particle.x, particle.y);
        } else {
          // Draw neural particles with glow effect
          ctx.beginPath();
          const gradient = ctx.createRadialGradient(
            particle.x, particle.y, 0,
            particle.x, particle.y, currentSize * 2
          );
          gradient.addColorStop(0, particle.color);
          gradient.addColorStop(1, 'rgba(100, 200, 255, 0)');
          
          ctx.fillStyle = gradient;
          ctx.arc(particle.x, particle.y, currentSize, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      animationRef.current = requestAnimationFrame(drawParticles);
    };

    initParticles();
    const animate = (time: number) => {
      timeRef.current = time;
      drawParticles(time);
    };
    animationRef.current = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener('resize', setCanvasSize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full z-0"
      style={{ background: 'transparent' }}
    />
  );
};

export default ParticlesComponent;
