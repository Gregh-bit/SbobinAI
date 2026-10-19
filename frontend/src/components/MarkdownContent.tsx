import { useMemo, type MouseEvent } from 'react';
import { marked } from 'marked';
import DOMPurify from 'dompurify';

interface MarkdownContentProps {
  content: string;
  className?: string;
}

// Configurazione standard marked
marked.setOptions({
  gfm: true,
  breaks: true,
});

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// I link esterni non devono far navigare via la webview Tauri: li apriamo in una nuova finestra
function handleLinkClick(e: MouseEvent<HTMLDivElement>) {
  const target = e.target as HTMLElement | null;
  const anchor = target?.closest('a');
  if (!anchor) return;
  const href = anchor.getAttribute('href') || '';
  if (/^https?:\/\//i.test(href)) {
    e.preventDefault();
    window.open(href, '_blank', 'noopener');
  }
}

export default function MarkdownContent({ content, className = '' }: MarkdownContentProps) {
  const html = useMemo(() => {
    if (!content) return '';
    try {
      return DOMPurify.sanitize(marked.parse(content) as string);
    } catch {
      return escapeHtml(content);
    }
  }, [content]);

  return (
    <div
      className={`markdown-body ${className}`}
      onClick={handleLinkClick}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
