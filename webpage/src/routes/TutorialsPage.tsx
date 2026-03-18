import { Link } from 'react-router-dom'

const tutorials = [
  {
    title: 'Binary Classification with MLP',
    duration: '20 min',
    description: 'Train and evaluate a small classifier end-to-end with NeuralCore.',
    doc: 'tutorial-binary-classification',
  },
  {
    title: 'Custom Loss Functions',
    duration: '15 min',
    description: 'Implement and debug custom differentiable objective functions.',
    doc: 'tutorial-custom-loss',
  },
  {
    title: 'Optimizer Benchmarking',
    duration: '25 min',
    description: 'Compare SGD, Adam, and RMSprop behavior on the same architecture.',
    doc: 'tutorial-optimizer-benchmarking',
  },
]

export default function TutorialsPage() {
  return (
    <section className="roadmap">
      <p className="section-label">Tutorials</p>
      <h1 className="docs-title">Hands-on Recipes</h1>
      <div className="timeline">
        {tutorials.map((tutorial) => (
          <Link key={tutorial.title} to={`/docs?doc=${tutorial.doc}`}>
            <h3>{tutorial.title}</h3>
            <p>{tutorial.description}</p>
            <p className="duration">{tutorial.duration}</p>
          </Link>
        ))}
      </div>
    </section>
  )
}
