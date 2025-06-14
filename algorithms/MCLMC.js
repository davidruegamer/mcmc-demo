"use strict";

MCMC.registerAlgorithm("MicrocanonicalLangevinMC", {
  description: "Microcanonical Langevin Monte Carlo (MCLMC)",

  about: () => {
    window.open("https://arxiv.org/pdf/2212.08549.pdf");
  },

  init: (self) => {
    self.leapfrogSteps = 40;
    self.dt = 0.15;
    self.eta = 0.1; // noise scale for partial momentum refreshment
  },

  reset: (self) => {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: (self, folder) => {
    folder.add(self, "leapfrogSteps", 5, 100).step(1).name("Leapfrog Steps");
    folder.add(self, "dt", 0.01, 0.5).step(0.01).name("Stepsize Δt");
    folder.add(self, "eta", 0.0, 1.0).step(0.01).name("Momentum noise η");
    folder.open();
  },

  step: (self, visualizer) => {

    const updateMomentum = function (eps, u, grad_logp) {
      const g_norm = Math.sqrt(grad_logp.norm2());
      if (g_norm < 1e-8) {
        // No meaningful gradient → slight random rotation
        const noise = MultivariateNormal.getSample(self.dim).scale(0.1);
        const u_new = u.add(noise);
        return u_new.scale(1 / Math.sqrt(u_new.norm2()));
      }

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
    let u = MultivariateNormal.getSample(self.dim);
    u = u.scale(1 / Math.sqrt(u.norm2())); // normalize initial direction

    const trajectory = [q.copy()];

    for (let i = 0; i < self.leapfrogSteps; i++) {
      // Half momentum update
      const grad = self.gradLogDensity(q);
      u = updateMomentum(self.dt / 2, u, grad);

      // Position update
      q.increment(u.scale(self.dt));

      // Half momentum update
      const grad2 = self.gradLogDensity(q);
      u = updateMomentum(self.dt / 2, u, grad2);

      // Partial momentum refreshment
      const z = MultivariateNormal.getSample(self.dim).scale(self.eta);
      const u_new = u.add(z);
      const norm_u_new = Math.sqrt(u_new.norm2());
      if (norm_u_new > 1e-12) {
        u = u_new.scale(1 / norm_u_new);
      }

      trajectory.push(q.copy());
    }

    // Visualize and accept
    visualizer.queue.push({
      type: "proposal",
      proposal: q,
      trajectory: trajectory,
      initialMomentum: u,
    });

    self.chain.push(q.copy());
    visualizer.queue.push({ type: "accept", proposal: q });
  },
});

