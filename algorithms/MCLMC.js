"use strict";

MCMC.registerAlgorithm("MicrocanonicalLangevinMC", {
  description: "Microcanonical Langevin Monte Carlo (MCLMC)",

  about: () => {
    window.open("https://arxiv.org/pdf/2212.08549.pdf");
  },

  init: (self) => {
    self.leapfrogSteps = 50;
    self.dt = 0.05;
    self.eta = 0.1;               // momentum noise scale
    self.energyThreshold = 0.2;   // max allowed energy deviation
  },

  reset: (self) => {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: (self, folder) => {
    folder.add(self, "leapfrogSteps", 5, 100).step(1).name("Leapfrog Steps");
    folder.add(self, "dt", 0.01, 0.5).step(0.01).name("Stepsize Δt");
    folder.add(self, "eta", 0.0, 1.0).step(0.01).name("Momentum Noise η");
    folder.add(self, "energyThreshold", 0.01, 1.0).step(0.01).name("Max ΔEnergy");
    folder.open();
  },

  step: (self, visualizer) => {

    const updateMomentum = function (eps, u, grad_logp) {
      const g_norm = Math.sqrt(grad_logp.norm2());
      if (g_norm < 1e-8) return u;

      const e = grad_logp.scale(-1 / g_norm);
      const ue = u.dot(e);
      const delta = eps * g_norm / (self.dim - 1);
      const zeta = Math.exp(-delta);
      const uu = e.scale((1 - zeta) * (1 + zeta + ue * (1 - zeta)))
                 .add(u.scale(2 * zeta));
      return uu.scale(1 / Math.sqrt(uu.norm2()));
    };

    const q0 = self.chain.last();
    let q = q0.copy();
    let u = MultivariateNormal.getSample(self.dim).scale(1 / Math.sqrt(self.dim));
    u = u.scale(1 / Math.sqrt(u.norm2())); // normalize to unit norm

    const U0 = -self.logDensity(q0);
    const trajectory = [q.copy()];

    for (let i = 0; i < self.leapfrogSteps; i++) {
      // Half momentum update
      u = updateMomentum(self.dt / 2, u, self.gradLogDensity(q));

      // Position update
      q.increment(u.scale(self.dt));

      // Half momentum update
      u = updateMomentum(self.dt / 2, u, self.gradLogDensity(q));

      // Orthogonal momentum refreshment
      const grad = self.gradLogDensity(q);
      const gradNorm = Math.sqrt(grad.norm2());
      if (gradNorm > 1e-8) {
        const e = grad.scale(1 / gradNorm);
        const z = MultivariateNormal.getSample(self.dim);
        const proj = z.subtract(e.scale(z.dot(e)));  // noise orthogonal to ∇logp
        const u_new = u.add(proj.scale(self.eta));
        const u_norm = Math.sqrt(u_new.norm2());
        if (u_norm > 1e-12) {
          u = u_new.scale(1 / u_norm);
        }
      }

      trajectory.push(q.copy());
    }

    const U1 = -self.logDensity(q);
    const deltaE = U1 - U0;

    visualizer.queue.push({
      type: "proposal",
      proposal: q,
      trajectory: trajectory,
      initialMomentum: u,
    });

    if (Math.abs(deltaE) < self.energyThreshold) {
      self.chain.push(q.copy());
      visualizer.queue.push({ type: "accept", proposal: q });
    } else {
      self.chain.push(q0.copy()); // reject, stay
      visualizer.queue.push({ type: "reject", proposal: q });
    }
  },
});

