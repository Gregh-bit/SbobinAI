import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

// Disabilita menu contestuale del browser (Reload, Ispeziona, ecc.) per una UX 100% nativa macOS
document.addEventListener('contextmenu', (e) => {
  e.preventDefault();
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)

