export default function PageHeader({ title, subtitle, children }) {
  return (
    <header className="mb-6 flex flex-col items-stretch justify-between gap-4 sm:mb-8 sm:flex-row sm:items-start">
      <div className="min-w-0 max-w-2xl">
        <h1 className="text-xl font-semibold uppercase tracking-wide text-white sm:text-2xl">{title}</h1>
        {subtitle ? <p className="mt-2 text-sm leading-relaxed text-sage-300">{subtitle}</p> : null}
      </div>
      {children ? <div className="flex shrink-0 flex-wrap items-center gap-3">{children}</div> : null}
    </header>
  );
}
