import { useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeSlug from 'rehype-slug'
import { docs, getSections } from '../content/docs'

export default function DocsPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const activeDocSlug = searchParams.get('doc') ?? docs[0].slug
  const filter = searchParams.get('q') ?? ''
  const normalizedFilter = filter.trim().toLowerCase()

  const activeDoc = useMemo(
    () => docs.find((doc) => doc.slug === activeDocSlug) ?? docs[0],
    [activeDocSlug],
  )

  const visibleDocs = useMemo(() => {
    if (!normalizedFilter) {
      return docs
    }

    return docs.filter((doc) => {
      const haystack = `${doc.title} ${doc.summary} ${doc.category}`.toLowerCase()
      return haystack.includes(normalizedFilter)
    })
  }, [normalizedFilter])

  const groupedDocs = useMemo(
    () => [
      {
        name: 'Guides',
        items: visibleDocs.filter((doc) => doc.category === 'Guide'),
      },
      {
        name: 'API',
        items: visibleDocs.filter((doc) => doc.category === 'API'),
      },
      {
        name: 'Tutorials',
        items: visibleDocs.filter((doc) => doc.category === 'Tutorial'),
      },
    ],
    [visibleDocs],
  )

  const sections = useMemo(() => getSections(activeDoc.content), [activeDoc.content])
  const filteredSections = useMemo(() => {
    if (!normalizedFilter) {
      return sections
    }

    return sections.filter((section) =>
      section.title.toLowerCase().includes(normalizedFilter),
    )
  }, [normalizedFilter, sections])

  function selectDoc(slug: string) {
    const next = new URLSearchParams(searchParams)
    next.set('doc', slug)
    setSearchParams(next)
  }

  function setQuery(value: string) {
    const next = new URLSearchParams(searchParams)
    if (value.trim()) {
      next.set('q', value)
    } else {
      next.delete('q')
    }
    setSearchParams(next)
  }

  return (
    <section className="docs-layout" aria-label="Documentation">
      <aside className="docs-sidebar">
        <p className="section-label">Documentation</p>
        <h1 className="docs-title">NeuralCore Guides</h1>

        <label className="docs-search-wrap" htmlFor="docs-search">
          <span>Search docs and sections</span>
          <input
            id="docs-search"
            className="docs-search"
            type="search"
            value={filter}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Try: backward, tensor, checkpoint"
          />
        </label>

        <nav aria-label="Doc pages" className="docs-nav">
          {groupedDocs.map((group) => (
            <div className="docs-nav-group" key={group.name}>
              <p>{group.name}</p>
              {group.items.length === 0 ? (
                <span className="empty">No entries</span>
              ) : (
                group.items.map((doc) => (
                  <button
                    className={doc.slug === activeDoc.slug ? 'active' : ''}
                    key={doc.slug}
                    onClick={() => selectDoc(doc.slug)}
                    type="button"
                  >
                    <strong>{doc.title}</strong>
                    <span>{doc.summary}</span>
                  </button>
                ))
              )}
            </div>
          ))}
        </nav>

        <div className="docs-sections">
          <p>Sections in this page</p>
          {filteredSections.length === 0 ? (
            <span className="empty">No matching sections.</span>
          ) : (
            filteredSections.map((section) => (
              <a
                className={section.level === 3 ? 'sub' : ''}
                href={`#${section.id}`}
                key={section.id}
              >
                {section.title}
              </a>
            ))
          )}
        </div>
      </aside>

      <article className="docs-content">
        <header className="docs-content-head">
          <p className="section-label">{activeDoc.category}</p>
          <h1>{activeDoc.title}</h1>
          <p className="docs-lead">{activeDoc.summary}</p>
          <div className="docs-meta">
            <span>{activeDoc.level}</span>
            <span>{activeDoc.minutes} min read</span>
            <span>Updated {activeDoc.updated}</span>
          </div>
          <div className="docs-prereq">
            <p>Prerequisites</p>
            <ul>
              {activeDoc.prerequisites.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        </header>

        <ReactMarkdown
          rehypePlugins={[rehypeSlug]}
          remarkPlugins={[remarkGfm]}
          components={{
            a: ({ href, ...props }) => {
              if (href?.startsWith('#')) {
                return <a href={href} {...props} />
              }

              return <a href={href} {...props} target="_blank" rel="noreferrer" />
            },
            h1: ({ ...props }) => <h2 {...props} />,
          }}
        >
          {activeDoc.content}
        </ReactMarkdown>
      </article>
    </section>
  )
}
