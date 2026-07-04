/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["Space Grotesk", "ui-sans-serif", "system-ui", "sans-serif"],
        sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "sans-serif"],
      },
      colors: {
        // Cold slate neutrals, not pure black, to avoid the "OLED off" look.
        surface: {
          950: "#0e1214",
          900: "#12181b",
          850: "#171d21",
          800: "#1f272c",
          700: "#2b353b",
          600: "#3a464e",
        },
        // Steel gray for passive text / informational states (kept name "sage").
        sage: {
          50: "#e7edf0",
          100: "#dbe3e7",
          200: "#c2ced4",
          300: "#a3b2ba",
          400: "#8fa1ab",
          500: "#6d7d86",
          600: "#55636b",
        },
        // Hazard amber, reserved for the primary action.
        accent: {
          200: "#ffdca6",
          300: "#ffc56b",
          400: "#ffb638",
          500: "#f5a524",
          600: "#d98c12",
        },
      },
      boxShadow: {
        soft: "0 20px 55px rgba(4, 8, 10, 0.55)",
        glow: "0 0 32px rgba(245, 165, 36, 0.22)",
      },
      letterSpacing: {
        label: "0.08em",
      },
    },
  },
  plugins: [],
};
