/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["Orbitron", "Chakra Petch", "ui-sans-serif", "system-ui", "sans-serif"],
        sans: ["Chakra Petch", "ui-sans-serif", "system-ui", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      colors: {
        // Near-black violet-tinted scale (was Geist gray; kept name "surface" to limit churn).
        surface: {
          950: "#07040d",
          900: "#0f0a1a",
          850: "#140d24",
          800: "#1a1230",
          700: "rgba(139,92,246,0.18)",
          600: "rgba(233,61,245,0.22)",
        },
        // Ink scale for text (kept name "sage" to limit churn).
        sage: {
          50: "#f4ecff",
          100: "#f4ecff",
          200: "#c9b8ea",
          300: "#a996c9",
          400: "#8a7ab3",
          500: "#6f5f8f",
          600: "#544a72",
        },
        // Amber retained strictly for warning states.
        accent: {
          200: "#ffd9a0",
          300: "#ffc266",
          400: "#ffb020",
          500: "#f5a623",
          600: "#d48806",
        },
        // High-tech neon accents.
        neon: {
          bg: "#07040d",
          panel: "#0f0a1a",
          magenta: "#f13df5",
          violet: "#8b5cf6",
          cyan: "#2fe8ea",
          green: "#3cf28a",
          red: "#ff3b6b",
          amber: "#ffb020",
        },
      },
      letterSpacing: {
        label: "0.08em",
      },
    },
  },
  plugins: [],
};
