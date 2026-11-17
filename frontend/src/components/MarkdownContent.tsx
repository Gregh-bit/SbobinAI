import { useMemo, type MouseEvent } from 'react';
import { Marked, type Tokens } from 'marked';
import DOMPurify from 'dompurify';
import katex from 'katex';
import 'katex/dist/katex.min.css';

interface MarkdownContentProps {
  content: string;
  className?: string;
  onOpenCitation?: (sourceId: string, page?: number) => void;
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

const markdownRenderer = new Marked({
  gfm: true,
  breaks: true,
  renderer: {
    code({ text, lang }: Tokens.Code): string {
      const cleanLang = (lang || '').trim().split(/\s+/)[0] || '';
      const langLabel = cleanLang || 'code';
      const langClass = cleanLang ? ` class="language-${escapeHtml(cleanLang)}"` : '';
      return (
        `<div class="code-block-wrapper">` +
        `<div class="code-block-header">` +
        `<span>${escapeHtml(langLabel)}</span>` +
        `<button type="button" class="code-copy-btn" data-code-copy="true">Copia</button>` +
        `</div>` +
        `<pre><code${langClass}>${escapeHtml(text)}</code></pre>` +
        `</div>`
      );
    },
  },
});

function renderKatexSafe(expr: string, displayMode: boolean): string {
  const rendered = katex.renderToString(expr.trim(), {
    displayMode,
    throwOnError: false,
    strict: false,
    trust: false,
  });
  return DOMPurify.sanitize(rendered, {
    USE_PROFILES: { html: true, mathMl: true },
    FORBID_TAGS: ['script', 'iframe', 'object', 'embed', 'form'],
    FORBID_ATTR: ['onerror', 'onload', 'onclick', 'onmouseover'],
  });
}

function parseMarkdownWithMathAndCitations(raw: string): string {
  const nonce = Math.random().toString(36).slice(2, 8);
  const codeTokenPrefix = `@@SBCODE_${nonce}_`;
  const katexTokenPrefix = `@@SBKATEX_${nonce}_`;
  const citeTokenPrefix = `@@SBCITE_${nonce}_`;

  const protectedCode: string[] = [];
  const katexTokens: string[] = [];
  const citationTokens: string[] = [];

  // 1. Protect fenced code blocks (including unclosed trailing blocks while streaming) and inline code spans
  let text = raw.replace(
    /(^|\n)([ \t]{0,3}(?:```|~~~)[^\n]*\n[\s\S]*?(?:\n[ \t]{0,3}(?:```|~~~)[ \t]*(?=\n|$)|$))/g,
    (_match, prefix: string, block: string) => {
      const idx = protectedCode.length;
      protectedCode.push(block);
      return `${prefix}${codeTokenPrefix}${idx}END`;
    }
  );

  text = text.replace(/`[^`\n]+`/g, (match) => {
    const idx = protectedCode.length;
    protectedCode.push(match);
    return `${codeTokenPrefix}${idx}END`;
  });

  // 2. Extract display math ($$...$$) and inline math ($...$) into safe KaTeX tokens
  text = text.replace(/\$\$([\s\S]+?)\$\$/g, (_match, expr: string) => {
    const idx = katexTokens.length;
    katexTokens.push(renderKatexSafe(expr, true));
    return `${katexTokenPrefix}${idx}ZTOKEN`;
  });

  text = text.replace(/(?<!\\)\$(?!\s)((?:\\.|[^$\n\\])+?)(?<!\s)\$/g, (_match, expr: string) => {
    const idx = katexTokens.length;
    katexTokens.push(renderKatexSafe(expr, false));
    return `${katexTokenPrefix}${idx}ZTOKEN`;
  });

  // 3. Extract clickable slide/material citations [[ref:SOURCE_ID#PAGE|LABEL]] or [[ref:SOURCE_ID|LABEL]]
  text = text.replace(/\[\[ref:([^\]|]+?)(?:#(\d+))?\|([^\]]+?)\]\]/g, (_match, sourceId: string, page: string | undefined, label: string) => {
    const idx = citationTokens.length;
    const safeSource = escapeHtml(sourceId.trim());
    const safeLabel = escapeHtml(label.trim());
    const pageAttr = page ? ` data-slide-page="${escapeHtml(page)}"` : '';
    citationTokens.push(
      `<a href="#slide-ref" data-slide-source="${safeSource}"${pageAttr} class="slide-citation-link underline decoration-emerald-400/70 underline-offset-4 text-emerald-300 hover:text-emerald-200 font-medium cursor-pointer transition-colors">${safeLabel}</a>`
    );
    return `${citeTokenPrefix}${idx}ZTOKEN`;
  });

  // 4. Restore protected code so marked parses code blocks normally
  const codeTokenRegex = new RegExp(`${codeTokenPrefix}(\\d+)END`, 'g');
  text = text.replace(codeTokenRegex, (_match, indexStr: string) => {
    return protectedCode[Number(indexStr)] ?? '';
  });

  // 5. Parse Markdown
  let html = markdownRenderer.parse(text) as string;

  // 6. Restore citation anchors before DOMPurify
  const citeTokenRegex = new RegExp(`${citeTokenPrefix}(\\d+)ZTOKEN`, 'g');
  html = html.replace(citeTokenRegex, (_match, indexStr: string) => {
    return citationTokens[Number(indexStr)] ?? '';
  });

  // 7. Sanitize HTML with custom citation & code-copy attributes allowed
  html = DOMPurify.sanitize(html, {
    ADD_ATTR: [
      'data-slide-source',
      'data-slide-page',
      'data-code-copy',
      'data-code-index',
      'target',
      'rel',
      'aria-hidden',
    ],
  });

  // 8. Restore sanitized KaTeX HTML (nonce prefix prevents collisions with literal text in code blocks)
  const katexTokenRegex = new RegExp(`${katexTokenPrefix}(\\d+)ZTOKEN`, 'g');
  html = html.replace(katexTokenRegex, (_match, indexStr: string) => {
    return katexTokens[Number(indexStr)] ?? '';
  });

  return html;
}

export default function MarkdownContent({
  content,
  className = '',
  onOpenCitation,
}: MarkdownContentProps) {
  const html = useMemo(() => {
    if (!content) return '';
    try {
      return parseMarkdownWithMathAndCitations(content);
    } catch {
      return escapeHtml(content);
    }
  }, [content]);

  function handleContainerClick(e: MouseEvent<HTMLDivElement>) {
    const target = e.target as HTMLElement | null;

    // 1. Clickable slide / material citation
    const citationEl = target?.closest('a[data-slide-source], button[data-slide-source]') as HTMLElement | null;
    if (citationEl) {
      e.preventDefault();
      const sourceId = citationEl.getAttribute('data-slide-source');
      if (sourceId) {
        const rawPage = citationEl.getAttribute('data-slide-page');
        const parsedPage = rawPage ? Number.parseInt(rawPage, 10) : NaN;
        onOpenCitation?.(sourceId, Number.isFinite(parsedPage) ? parsedPage : undefined);
      }
      return;
    }

    // 2. 1-click code block copy button
    const copyBtn = target?.closest('button[data-code-copy]') as HTMLButtonElement | null;
    if (copyBtn) {
      e.preventDefault();
      const wrapper = copyBtn.closest('.code-block-wrapper');
      const codeText = wrapper?.querySelector('pre code')?.textContent ?? '';
      if (!codeText) return;
      navigator.clipboard
        ?.writeText(codeText)
        .then(() => {
          copyBtn.textContent = 'Copiato';
          window.setTimeout(() => {
            if (copyBtn.isConnected) {
              copyBtn.textContent = 'Copia';
            }
          }, 1500);
        })
        .catch(() => {});
      return;
    }

    // 3. External links open in new window so Tauri webview doesn't navigate away
    const anchor = target?.closest('a');
    if (!anchor) return;
    const href = anchor.getAttribute('href') || '';
    if (/^https?:\/\//i.test(href)) {
      e.preventDefault();
      window.open(href, '_blank', 'noopener,noreferrer');
    }
  }

  return (
    <div
      className={`markdown-body ${className}`}
      onClick={handleContainerClick}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
