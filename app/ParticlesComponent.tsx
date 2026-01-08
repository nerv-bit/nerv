"use client";

import { ParticlesBg } from 'particle-bg';

const ParticlesComponent = () => {
  return (
    <ParticlesBg
      type="cobweb"
      bg={true}
      color="#ffffff"
      num={100}
      config={{
        num: [4, 7],
        rps: 0.1,
        radius: [5, 40],
        life: [1.5, 3],
        v: [0.5, 0.8],
        tha: [-40, 40],
        alpha: [0.6, 0],
        scale: [0.1, 0.4],
        position: "all",
        color: ["#ffffff", "#4dc3ff"],
        cross: "dead",
        random: 15
      }}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 0,
      }}
    />
  );
};

export default ParticlesComponent;
