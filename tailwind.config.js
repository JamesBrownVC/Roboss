/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["Geist", "ui-sans-serif", "system-ui", "sans-serif"],
        sans: ["Geist", "ui-sans-serif", "system-ui", "Segoe UI", "sans-serif"],
        mono: ["Geist Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      colors: {
        // Geist dark scale: pure black page, near-black surfaces, 1px gray borders.
        surface: {
          950: "#000000",
          900: "#0a0a0a",
          850: "#111111",
          800: "#1a1a1a",
          700: "#262626",
          600: "#333333",
        },
        // Geist grays for text (kept name "sage" to limit churn).
        sage: {
          50: "#fafafa",
          100: "#ededed",
          200: "#d4d4d4",
          300: "#a1a1a1",
          400: "#8f8f8f",
          500: "#666666",
          600: "#444444",
        },
        // Amber retained strictly for warning states (Geist amber).
        accent: {
          200: "#ffe3a3",
          300: "#ffcb6b",
          400: "#ffb224",
          500: "#f5a623",
          600: "#d48806",
        },
      },
      letterSpacing: {
        label: "0.08em",
      },
    },
  },
  plugins: [],
};
