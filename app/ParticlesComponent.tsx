'use client';

import { useCallback } from "react";
import Particles from "@tsparticles/react";
import { loadSlim } from "@tsparticles/slim"; // or loadFull if you need more features
// If using a different package, adjust import accordingly

const ParticlesComponent = () => {
  const particlesInit = useCallback(async (engine) => {
    await loadSlim(engine); // or loadFull(engine)
  }, []);

  return (
    <Particles
      id="tsparticles"
      init={particlesInit}
      options={{
        // Your particles config options here (paste from your current setup)
        background: { color: { value: "#000000" } },
        fpsLimit: 120,
        // ... rest of your options
      }}
    />
  );
};

export default ParticlesComponent;
