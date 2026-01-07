'use client';

import { useCallback, useMemo } from "react";
import Particles from "@tsparticles/react";

const ParticlesComponent = () => {
  const particlesInit = useCallback(async (engine: any) => {
    const { loadSlim } = await import("@tsparticles/slim");
    await loadSlim(engine);
  }, []);

  const particlesOptions = useMemo(
    () => ({
      background: {
        color: {
          value: '#000000',
        },
      },
      fpsLimit: 120,
      interactivity: {
        events: {
          onHover: {
            enable: true,
            mode: 'repulse',
          },
          resize: { enable: true },
        },
        modes: {
          repulse: {
            distance: 150,
            duration: 0.4,
          },
        },
      },
      particles: {
        color: {
          value: ['#00ffff', '#8a2be2', '#ffffff'],
        },
        links: {
          color: '#00ffff',
          distance: 180,
          enable: true,
          opacity: 0.4,
          width: 1,
        },
        move: {
          enable: true,
          speed: 0.8,
          direction: 'none',
          random: false,
          straight: false,
          outModes: {
            default: 'bounce',
          },
        },
        number: {
          density: {
            enable: true,
            area: 1000,
          },
          value: 100,
        },
        opacity: {
          value: { min: 0.1, max: 0.6 },
          random: true,
          animation: {
            enable: true,
            speed: 1,
            minimumValue: 0.1,
            sync: false,
          },
        },
        shape: {
          type: 'circle',
        },
        size: {
          value: { min: 1, max: 4 },
          random: true,
        },
      },
      detectRetina: true,
    }),
    []
  );

  return (
    <Particles
      id="tsparticles"
      init={particlesInit}
      options={particlesOptions}
      className="absolute inset-0 -z-10"
    />
  );
};

export default ParticlesComponent;
