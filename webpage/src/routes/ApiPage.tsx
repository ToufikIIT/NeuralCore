import { Link } from 'react-router-dom'

const modules = [
  {
    name: 'Tensor',
    description: 'N-dimensional tensor creation, indexing, math ops, and reductions.',
    status: 'Stable',
    doc: 'api-tensor',
  },
  {
    name: 'Autograd',
    description: 'Dynamic computation graph with automatic differentiation support.',
    status: 'Stable',
    doc: 'api-autograd',
  },
  {
    name: 'NN',
    description: 'Layers, activations, losses, and module composition utilities.',
    status: 'Stable',
    doc: 'api-nn',
  },
  {
    name: 'Optim',
    description: 'SGD, Adam, RMSprop, and custom optimizer parameter groups.',
    status: 'Stable',
    doc: 'api-optim',
  },
]

export default function ApiPage() {
  return (
    <section className="api">
      <div className="api-head">
        <div>
          <p className="section-label">API Reference</p>
          <h1 className="docs-title">Modules and Namespaces</h1>
        </div>
      </div>
      <div className="api-grid">
        {modules.map((module) => (
          <Link className="api-card" key={module.name} to={`/docs?doc=${module.doc}`}>
            <h3>{module.name}</h3>
            <p>{module.description}</p>
            <span className="status-pill">{module.status}</span>
          </Link>
        ))}
      </div>
    </section>
  )
}
