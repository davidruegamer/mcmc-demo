"use strict";

MCMC.registerAlgorithm("MicrocanonicalHamiltonianMC", {
  description: "Microcanonical Hamiltonian Monte Carlo",

  about: () => {
    window.open("https://arxiv.org/pdf/2212.08549.pdf");
  },

  init: (self) => {
    self.leapfrogSteps = 40;
    self.dt = 0.15;
  },

  reset: (self) => {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: (self, folder) => {
    folder.add(self, "leapfrogSteps", 5, 100).step(1).name("Leapfrog Steps");
    folder.add(self, "dt", 0.01, 0.5).step(0.01).name("Leapfrog Δt");
    folder.open();
  },

  step: (self, visualizer) => {
    const q0 = self.chain.last();
    const p0 = MultivariateNormal.getSample(self.dim);

    // Compute initial energy
    const kinetic = 0.5 * p0.norm2();
    const potential = -self.logDensity(q0);
    const E = kinetic + potential;

    let q = q0.copy();
    let p = p0.copy();
    const trajectory = [q.copy()];

    for (let i = 0; i < self.leapfrogSteps; i++) {
      // Half step for momentum
      const gradU = self.gradLogDensity(q).scale(-1); // grad of potential
      const gradNorm = Math.sqrt(gradU.norm2());
      if (gradNorm > 1e-10) {
        const e = gradU.scale(1 / gradNorm);
        const pDotE = p.dot(e);
        const pTangent = p.subtract(e.scale(pDotE)); // Project onto tangent space
        p = pTangent; // Momentum becomes tangential
      }

      p.increment(self.gradLogDensity(q).scale(self.dt / 2)); // tangential push
      // Full step for position
      q.increment(p.scale(self.dt));
      // Half step for momentum
      const gradU2 = self.gradLogDensity(q).scale(-1);
      const gradNorm2 = Math.sqrt(gradU2.norm2());
      if (gradNorm2 > 1e-10) {
        const e2 = gradU2.scale(1 / gradNorm2);
        const pDotE2 = p.dot(e2);
        const pTangent2 = p.subtract(e2.scale(pDotE2));
        p = pTangent2;
      }

      p.increment(self.gradLogDensity(q).scale(self.dt / 2));
      trajectory.push(q.copy());
    }

    // Renormalize momentum to preserve total energy
    const U_new = -self.logDensity(q);
    const p_new_norm = Math.sqrt(2 * Math.max(E - U_new, 1e-10));
    const p_norm_current = Math.sqrt(p.norm2());
    if (p_norm_current > 1e-10) {
      p = p.scale(p_new_norm / p_norm_current);
    }

    visualizer.queue.push({
      type: "proposal",
      proposal: q,
      trajectory: trajectory,
      initialMomentum: p0,
    });

    // Deterministic proposal (no Metropolis step for this integrator)
    self.chain.push(q.copy());
    visualizer.queue.push({ type: "accept", proposal: q });
  },
});

