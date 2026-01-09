"use client";
import { useEffect, useRef } from 'react';

const NeuralBackground = ({ isActive }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animationId;
    let particles = [];

    // Initialize canvas size
    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      initParticles();
    };

    const initParticles = () => {
      particles = [];
      // INCREASED PARTICLE COUNT for more density
      const count = Math.min(80, (window.innerWidth * window.innerHeight) / 8000); // Was 40, 15000
      for (let i = 0; i < count; i++) {
        particles.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          size: Math.random() * 3 + 1.5, // INCREASED SIZE (was 1.5 + 0.5)
          speedX: (Math.random() - 0.5) * 0.5, // Slightly faster
          speedY: (Math.random() - 0.5) * 0.5,
          // BRIGHTER COLORS with higher opacity
          color: isActive 
            ? `rgba(100, 220, 255, ${0.4 + Math.random() * 0.4})` // Active: very bright
            : `rgba(100, 200, 255, ${0.2 + Math.random() * 0.3})`, // Inactive: still visible
          pulseOffset: Math.random() * Math.PI * 2,
        });
      }
    };

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw connections with ENHANCED VISIBILITY
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          
          // INCREASED CONNECTION RANGE and visibility
          if (dist < 180) { // Was 120
            const baseAlpha = isActive ? 0.4 : 0.15; // Much more visible
            const alpha = baseAlpha * (1 - dist / 180);
            
            ctx.beginPath();
            ctx.strokeStyle = isActive 
              ? `rgba(100, 240, 255, ${alpha})` // Brighter blue when active
              : `rgba(100, 200, 255, ${alpha})`;
            ctx.lineWidth = isActive ? 1.2 : 0.8; // Thicker lines
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }
      
      // Update & draw particles with ENHANCED EFFECTS
      const time = Date.now() * 0.001;
      
      particles.forEach(p => {
        // Pulsing effect for "living system" feel
        const pulse = isActive 
          ? (Math.sin(time * 2 + p.pulseOffset) + 1) * 0.3 // Stronger pulse when active
          : (Math.sin(time * 0.5 + p.pulseOffset) + 1) * 0.1;
        
        const currentSize = p.size + pulse * 2; // Larger pulsing
        
        // Neural wave movement when active
        if (isActive) {
          p.x += p.speedX + Math.sin(time + p.y * 0.01) * 0.2;
          p.y += p.speedY + Math.cos(time + p.x * 0.01) * 0.2;
        } else {
          p.x += p.speedX;
          p.y += p.speedY;
        }
        
        // Boundary check with bounce
        if (p.x < 0 || p.x > canvas.width) p.speedX *= -1;
        if (p.y < 0 || p.y > canvas.height) p.speedY *= -1;
        
        // Draw particle with GLOW EFFECT
        ctx.beginPath();
        
        // Create gradient for glow effect (more visible)
        const gradient = ctx.createRadialGradient(
          p.x, p.y, 0,
          p.x, p.y, currentSize * 2
        );
        
        if (isActive) {
          gradient.addColorStop(0, `rgba(150, 240, 255, 0.9)`); // Bright center
          gradient.addColorStop(0.5, `rgba(100, 220, 255, 0.4)`); // Middle
          gradient.addColorStop(1, `rgba(50, 150, 255, 0)`); // Outer edge
        } else {
          gradient.addColorStop(0, `rgba(100, 200, 255, 0.7)`);
          gradient.addColorStop(0.7, `rgba(70, 150, 255, 0.2)`);
          gradient.addColorStop(1, `rgba(30, 100, 255, 0)`);
        }
        
        ctx.fillStyle = gradient;
        ctx.arc(p.x, p.y, currentSize * 1.5, 0, Math.PI * 2); // Larger glow area
        ctx.fill();
        
        // Core particle (even brighter)
        ctx.beginPath();
        ctx.fillStyle = isActive 
          ? `rgba(200, 240, 255, 0.9)` 
          : `rgba(150, 220, 255, 0.7)`;
        ctx.arc(p.x, p.y, currentSize * 0.7, 0, Math.PI * 2);
        ctx.fill();
      });
      
      animationId = requestAnimationFrame(animate);
    };

    window.addEventListener('resize', resize);
    resize();
    animate();

    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animationId);
    };
  }, [isActive]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 w-screen h-screen z-0 pointer-events-none"
      style={{ 
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        // HIGHER OPACITY for better visibility
        opacity: isActive ? 1 : 0.4, // Was 0.8 and 0.2
        transition: 'opacity 0.8s ease',
        // Optional: Add a subtle background gradient
        background: 'radial-gradient(ellipse at center, rgba(10, 20, 40, 0.2) 0%, rgba(0, 0, 0, 0.8) 100%)'
      }}
    />
  );
};
export default NeuralBackground;
