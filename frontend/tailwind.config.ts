import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}", // make sure it scans TS & TSX
  ],
  theme: {
    extend: {
      fontFamily: {
        epunda: ["Epunda Slab", "sans-serif"],
        kanit: ["Kanit", "sans-serif"],
        chiron: ["Chiron GoRound TC", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;



