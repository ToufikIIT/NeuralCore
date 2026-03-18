import { Link } from 'react-router-dom'

const guides = [
  {
    title: 'Getting Started',
    detail: 'Install NeuralCore and train your first network in under 10 minutes.',
    tag: 'Beginner',
    to: '/docs?doc=getting-started',
  },
  {
    title: 'Tensor Fundamentals',
    detail: 'Shapes, broadcasting, slicing, and low-level tensor memory behavior.',
    tag: 'Core',
    to: '/docs?doc=tensor-fundamentals',
  },
  {
    title: 'Autograd Deep Dive',
    detail: 'Understand graph construction, backward passes, and gradient inspection.',
    tag: 'Intermediate',
    to: '/docs?doc=autograd-deep-dive',
  },
  {
    title: 'Training at Scale',
    detail: 'Build robust training loops with optimizers, checkpointing, and metrics.',
    tag: 'Advanced',
    to: '/docs?doc=training-at-scale',
  },
]

const quickStartCode = `#include <neuralcore/neuralcore.hpp>

using namespace nc;

int main() {
  Tensor x = Tensor::randn({64, 2});
  Tensor y = Tensor::randn({64, 1});

  nn::Sequential model({
    nn::Linear(2, 32),
    nn::ReLU(),
    nn::Linear(32, 1)
  });

  optim::Adam optim(model.parameters(), 1e-3);

  for (int step = 0; step < 1000; ++step) {
    Tensor pred = model.forward(x);
    Tensor loss = nn::mse_loss(pred, y);
    optim.zero_grad();
    loss.backward();
    optim.step();
  }
}`

export default function HomePage() {
  return (
    <>
      <section className="hero">
        <p className="eyebrow">Version 0.1.0</p>
        <h1>Build fast models with a clean C++ API inspired by PyTorch.</h1>
        <p className="hero-copy">
          Documentation-first design, tensor primitives, automatic differentiation,
          and composable neural network modules in one compact framework.
        </p>
        <div className="hero-actions">
          <Link className="solid" to="/docs">
            Read the Docs
          </Link>
          <Link className="ghost" to="/api">
            Explore API
          </Link>
        </div>
        <div className="signal-grid" role="presentation" aria-hidden="true">
          <span>Tensor</span>
          <span>Autograd</span>
          <span>NN</span>
          <span>Optim</span>
        </div>
      </section>

      <section className="guide-grid">
        {guides.map((guide) => (
          <Link className="guide-card" to={guide.to} key={guide.title}>
            <p className="guide-tag">{guide.tag}</p>
            <h2>{guide.title}</h2>
            <p>{guide.detail}</p>
          </Link>
        ))}
      </section>

      <section className="quickstart">
        <div>
          <p className="section-label">Quickstart</p>
          <h2>From tensors to trained model in one page.</h2>
          <p>
            This starter flow mirrors your framework goals: define modules,
            compute loss, call backward, and update with Adam.
          </p>
        </div>
        <pre>
          <code>{quickStartCode}</code>
        </pre>
      </section>
    </>
  )
}
