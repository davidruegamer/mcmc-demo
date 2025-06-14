"use strict";

MCMC.registerAlgorithm("MicrocanonicalLangevinMC", {
  description: "Microcanonical Langevin Monte Carlo (MCLMC)",

  about: () => {
    window.open("https://arxiv.org/abs/2303.18221");
  },

  init: (self) => {
    self.leapfrogSteps = 40;
    self.dt = 0.15;
    self.eta = 0.1; // Refresh strength
  },

  reset: (self) => {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: (self, folder) => {
    folder.add(self, "leapfrogSteps", 5, 100).step(1).name("Leapfrog Steps");
    folder.add(self, "dt", 0.01, 0.5).step(0.01).name("Leapfrog Δt");
    folder.add(self, "eta", 0.0, 1.0).step(0.01).name("Noise Strength η");
    folder.open();
  },

  step: (self, visualizer) => {
    const q0 = self.chain.last();
    const p0 = MultivariateNormal.getSample(self.dim);

    const kinetic = 0.5 * p0.norm2();
    const potential = -self.logDensity(q0);
    const E = kinetic + potential;

    let q = q0.copy();
    let p = p0.copy();
    const trajectory = [q.copy()];

    for (let i = 0; i < self.leapfrogSteps; i++) {
      const gradU = self.gradLogDensity(q).scale(-1);
      const gradNorm = Math.sqrt(gradU.norm2());

      if (gradNorm > 1e-10) {
        const e = gradU.scale(1 / gradNorm);
        const pDotE = p.dot(e);
        let pTangent = p.subtract(e.scale(pDotE)); // project

        // Partial stochastic refresh in tangent plane
        const z = MultivariateNormal.getSample(self.dim);
        const zTangent = z.subtract(e.scale(z.dot(e)));
        pTangent = pTangent.add(zTangent.scale(self.eta));

        // Normalize to original momentum norm
        const newNorm = Math.sqrt(pTangent.norm2());
        if (newNorm > 1e-10) {
          p = pTangent.scale(Math.sqrt(p.norm2()) / newNorm);
        }
      }

      p.increment(self.gradLogDensity(q).scale(self.dt / 2));
      q.increment(p.scale(self.dt));
      const gradU2 = self.gradLogDensity(q).scale(-1);
      const gradNorm2 = Math.sqrt(gradU2.norm2());

      if (gradNorm2 > 1e-10) {
        const e2 = gradU2.scale(1 / gradNorm2);
        const pDotE2 = p.dot(e2);
        let pTangent2 = p.subtract(e2.scale(pDotE2));

        // Partial refresh again
        const z2 = MultivariateNormal.getSample(self.dim);
        const zTangent2 = z2.subtract(e2.scale(z2.dot(e2)));
        pTangent2 = pTangent2.add(zTangent2.scale(self.eta));

        const newNorm2 = Math.sqrt(pTangent2.norm2());
        if (newNorm2 > 1e-10) {
          p = pTangent2.scale(Math.sqrt(p.norm2()) / newNorm2);
        }
      }

      p.increment(self.gradLogDensity(q).scale(self.dt / 2));
      trajectory.push(q.copy());
    }

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

    self.chain.push(q.copy());
    visualizer.queue.push({ type: "accept", proposal: q });
  },
});

