/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: {
          DEFAULT: '#0a0c10',
          darker: '#07080b',
        },
        glass: {
          DEFAULT: 'rgba(255, 255, 255, 0.04)',
          hover: 'rgba(255, 255, 255, 0.07)',
          active: 'rgba(255, 255, 255, 0.10)',
          pill: 'rgba(255, 255, 255, 0.06)',
          border: 'rgba(255, 255, 255, 0.11)',
          borderSubtle: 'rgba(255, 255, 255, 0.06)',
          borderHover: 'rgba(255, 255, 255, 0.20)',
        },
        traffic: {
          close: '#ff5f56',
          minimize: '#ffbd2e',
          zoom: '#27c93f',
        },
      },
      fontFamily: {
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          '"SF Pro Display"',
          '"SF Pro Text"',
          '"SF Pro"',
          'system-ui',
          'sans-serif',
        ],
        mono: [
          '"SF Mono"',
          'Menlo',
          'Monaco',
          'ui-monospace',
          'monospace',
        ],
      },
      letterSpacing: {
        tighter: '-0.03em',
        tight: '-0.015em',
      },
      boxShadow: {
        'liquid': '0 8px 32px 0 rgba(0, 0, 0, 0.36), inset 0 1px 0 0 rgba(255, 255, 255, 0.15)',
        'liquid-sm': '0 4px 16px 0 rgba(0, 0, 0, 0.24), inset 0 1px 0 0 rgba(255, 255, 255, 0.12)',
        'liquid-md': '0 12px 40px 0 rgba(0, 0, 0, 0.42), inset 0 1px 0 0 rgba(255, 255, 255, 0.16)',
        'liquid-lg': '0 20px 60px 0 rgba(0, 0, 0, 0.5), inset 0 1px 0 0 rgba(255, 255, 255, 0.2)',
        'liquid-pill': '0 2px 10px 0 rgba(0, 0, 0, 0.2), inset 0 1px 0 0 rgba(255, 255, 255, 0.18)',
        'capsule': '0 24px 48px -8px rgba(0, 0, 0, 0.7), 0 1px 2px rgba(255, 255, 255, 0.1), inset 0 1px 0 0 rgba(255, 255, 255, 0.2)',
      },
      animation: {
        'fade-in': 'fadeIn 0.18s cubic-bezier(0.16, 1, 0.3, 1)',
        'slide-up': 'slideUp 0.24s cubic-bezier(0.16, 1, 0.3, 1)',
        'slide-down': 'slideDown 0.25s cubic-bezier(0.16, 1, 0.3, 1) forwards',
        'scale-in': 'scaleIn 0.2s cubic-bezier(0.16, 1, 0.3, 1) forwards',
        'pulse-subtle': 'pulseSubtle 2.5s ease-in-out infinite',
        'wave': 'wave 1.2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideDown: {
          '0%': { opacity: '0', transform: 'translateY(-8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        scaleIn: {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        pulseSubtle: {
          '0%, 100%': { opacity: '0.4' },
          '50%': { opacity: '0.9' },
        },
        wave: {
          '0%, 100%': { height: '6px' },
          '50%': { height: '18px' },
        },
      },
    },
  },
  plugins: [],
}
