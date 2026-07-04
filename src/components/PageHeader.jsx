export default function PageHeader({ title, subtitle, children }) {
  return (
    <header className="mb-8 flex flex-wrap items-start justify-between gap-4">
      <div className="max-w-2xl">
        <h1 className="font-display text-3xl font-bold tracking-tight text-white">{title}</h1>
        {subtitle ? <p className="mt-2 text-[15px] leading-relaxed text-sage-200/90">{subtitle}</p> : null}
      </div>
      {children}
    </header>
  );
}
