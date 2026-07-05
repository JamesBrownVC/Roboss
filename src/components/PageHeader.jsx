export default function PageHeader({ title, subtitle, children }) {
  return (
    <header className="mb-8 flex flex-wrap items-start justify-between gap-4">
      <div className="max-w-2xl">
        <h1 className="text-2xl font-semibold uppercase tracking-wide text-white">{title}</h1>
        {subtitle ? <p className="mt-2 text-sm leading-relaxed text-sage-300">{subtitle}</p> : null}
      </div>
      {children}
    </header>
  );
}
