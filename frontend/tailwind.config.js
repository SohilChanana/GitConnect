/** @type {import('tailwindcss').Config} */
export default {
    darkMode: ["class"],
    content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
  	extend: {
  		colors: {
        background: '#09090b', // Zinc 950
        foreground: '#fafafa', // Zinc 50
        card: {
          DEFAULT: '#18181b', // Zinc 900
          foreground: '#fafafa'
        },
        popover: {
          DEFAULT: '#18181b',
          foreground: '#fafafa'
        },
        primary: {
          DEFAULT: '#8b5cf6', // Violet 500
          foreground: '#ffffff'
        },
        secondary: {
          DEFAULT: '#27272a', // Zinc 800
          foreground: '#fafafa'
        },
        muted: {
          DEFAULT: '#27272a',
          foreground: '#a1a1aa' // Zinc 400
        },
        accent: {
          DEFAULT: '#27272a',
          foreground: '#fafafa'
        },
        destructive: {
          DEFAULT: '#7f1d1d',
          foreground: '#fef2f2'
        },
        border: '#27272a',
        input: '#27272a',
        ring: '#d4d4d8',
  		},
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      borderRadius: {
  			lg: 'var(--radius)',
  			md: 'calc(var(--radius) - 2px)',
  			sm: 'calc(var(--radius) - 4px)'
  		}
  	}
  },
  plugins: [],
}
