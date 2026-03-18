import gettingStarted from './markdown/getting-started.md?raw'
import tensorFundamentals from './markdown/tensor-fundamentals.md?raw'
import autogradDeepDive from './markdown/autograd-deep-dive.md?raw'
import trainingScale from './markdown/training-at-scale.md?raw'
import apiTensor from './markdown/api-tensor.md?raw'
import apiAutograd from './markdown/api-autograd.md?raw'
import apiNn from './markdown/api-nn.md?raw'
import apiOptim from './markdown/api-optim.md?raw'
import tutorialBinary from './markdown/tutorial-binary-classification.md?raw'
import tutorialCustomLoss from './markdown/tutorial-custom-loss.md?raw'
import tutorialOptimizerBench from './markdown/tutorial-optimizer-benchmarking.md?raw'

export type DocEntry = {
  slug: string
  title: string
  summary: string
  content: string
  category: 'Guide' | 'API' | 'Tutorial'
  level: 'Beginner' | 'Intermediate' | 'Advanced'
  minutes: number
  updated: string
  prerequisites: string[]
}

export type DocSection = {
  id: string
  title: string
  level: number
}

export const docs: DocEntry[] = [
  {
    slug: 'getting-started',
    title: 'Getting Started',
    summary: 'Install NeuralCore and train your first model in a few minutes.',
    content: gettingStarted,
    category: 'Guide',
    level: 'Beginner',
    minutes: 12,
    updated: '2026-03-18',
    prerequisites: ['CMake 3.20+', 'C++17 compiler', 'Basic linear algebra'],
  },
  {
    slug: 'tensor-fundamentals',
    title: 'Tensor Fundamentals',
    summary: 'Understand shapes, broadcasting, indexing, and common tensor ops.',
    content: tensorFundamentals,
    category: 'Guide',
    level: 'Beginner',
    minutes: 18,
    updated: '2026-03-18',
    prerequisites: ['Getting Started', 'Array programming basics'],
  },
  {
    slug: 'autograd-deep-dive',
    title: 'Autograd Deep Dive',
    summary: 'How computation graphs and backpropagation work under the hood.',
    content: autogradDeepDive,
    category: 'Guide',
    level: 'Intermediate',
    minutes: 22,
    updated: '2026-03-18',
    prerequisites: ['Tensor Fundamentals', 'Calculus basics'],
  },
  {
    slug: 'training-at-scale',
    title: 'Training at Scale',
    summary: 'Build robust training loops with checkpoints and metrics.',
    content: trainingScale,
    category: 'Guide',
    level: 'Advanced',
    minutes: 24,
    updated: '2026-03-18',
    prerequisites: ['Autograd Deep Dive', 'Model evaluation metrics'],
  },
  {
    slug: 'api-tensor',
    title: 'Tensor API Reference',
    summary: 'Core tensor constructors, operators, and indexing patterns.',
    content: apiTensor,
    category: 'API',
    level: 'Beginner',
    minutes: 15,
    updated: '2026-03-18',
    prerequisites: ['Tensor Fundamentals'],
  },
  {
    slug: 'api-autograd',
    title: 'Autograd API Reference',
    summary: 'Gradient control, graph life-cycle, and backward semantics.',
    content: apiAutograd,
    category: 'API',
    level: 'Intermediate',
    minutes: 14,
    updated: '2026-03-18',
    prerequisites: ['Autograd Deep Dive'],
  },
  {
    slug: 'api-nn',
    title: 'NN API Reference',
    summary: 'Layers, module composition, and training-time behavior.',
    content: apiNn,
    category: 'API',
    level: 'Intermediate',
    minutes: 16,
    updated: '2026-03-18',
    prerequisites: ['Getting Started'],
  },
  {
    slug: 'api-optim',
    title: 'Optim API Reference',
    summary: 'Optimizer construction, stepping, and parameter group tuning.',
    content: apiOptim,
    category: 'API',
    level: 'Intermediate',
    minutes: 13,
    updated: '2026-03-18',
    prerequisites: ['Training at Scale'],
  },
  {
    slug: 'tutorial-binary-classification',
    title: 'Tutorial: Binary Classification with MLP',
    summary: 'Build a complete classifier with proper train and validation loops.',
    content: tutorialBinary,
    category: 'Tutorial',
    level: 'Beginner',
    minutes: 20,
    updated: '2026-03-18',
    prerequisites: ['Getting Started', 'Tensor Fundamentals'],
  },
  {
    slug: 'tutorial-custom-loss',
    title: 'Tutorial: Custom Loss Functions',
    summary: 'Design, implement, and validate custom objective functions.',
    content: tutorialCustomLoss,
    category: 'Tutorial',
    level: 'Advanced',
    minutes: 18,
    updated: '2026-03-18',
    prerequisites: ['Autograd Deep Dive', 'C++ templates basics'],
  },
  {
    slug: 'tutorial-optimizer-benchmarking',
    title: 'Tutorial: Optimizer Benchmarking',
    summary: 'Benchmark convergence and stability across SGD, Adam, and RMSprop.',
    content: tutorialOptimizerBench,
    category: 'Tutorial',
    level: 'Advanced',
    minutes: 25,
    updated: '2026-03-18',
    prerequisites: ['Training at Scale', 'Metric tracking'],
  },
]

const headingToId = (value: string): string =>
  value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9\s-]/g, '')
    .replace(/\s+/g, '-')

export function getSections(content: string): DocSection[] {
  const sections: DocSection[] = []
  const matcher = /^(##|###)\s+(.+)$/gm
  let match = matcher.exec(content)

  while (match) {
    const level = match[1] === '##' ? 2 : 3
    const title = match[2].trim()
    sections.push({ id: headingToId(title), title, level })
    match = matcher.exec(content)
  }

  return sections
}
