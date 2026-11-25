import { memo, useMemo, type MouseEvent } from 'react';
import { Marked, type Tokens } from 'marked';
import DOMPurify from 'dompurify';
import katex from 'katex';
import 'katex/dist/katex.min.css';

interface MarkdownContentProps {
  content: string;
  className?: string;
  onOpenCitation?: (sourceId: string, page?: number) => void;
  onOpenAudioSnippet?: (lectureId: number, seconds: number, autoPlay?: boolean) => void;
  onOpenSettings?: () => void;
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

function parseTimestampToSeconds(val: string | undefined): number {
  if (!val) return 0;
  const trimmed = val.trim();
  if (trimmed.includes(':')) {
    const parts = trimmed.split(':').map((p) => Number.parseFloat(p));
    if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
      return Math.max(0, parts[0] * 60 + parts[1]);
    }
    if (parts.length === 3 && !isNaN(parts[0]) && !isNaN(parts[1]) && !isNaN(parts[2])) {
      return Math.max(0, parts[0] * 3600 + parts[1] * 60 + parts[2]);
    }
  }
  const num = Number.parseFloat(trimmed);
  return isNaN(num) ? 0 : Math.max(0, num);
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

  // 3a. Extract clickable audio citations [[audio:LECTURE_ID|SECONDS|LABEL]] or [[audio:LECTURE_ID#SECONDS|LABEL]]
  text = text.replace(
    /\[\[audio:(\d+)(?:[#|](\d+(?::\d+)?(?:\.\d+)?))?\|([^\]]+?)\]\]/g,
    (_match, lectureId: string, secondsStr: string | undefined, label: string) => {
      const idx = citationTokens.length;
      const safeLecId = escapeHtml(lectureId.trim());
      const parsedSeconds = parseTimestampToSeconds(secondsStr);
      const safeSec = escapeHtml(String(parsedSeconds));
      const safeLabel = escapeHtml(label.trim());
      citationTokens.push(
        `<a href="#audio-ref" data-audio-lecture-id="${safeLecId}" data-audio-seconds="${safeSec}" class="audio-citation-link inline-flex items-center gap-1 font-medium text-sky-300 hover:text-sky-100 underline decoration-sky-400/60 hover:decoration-sky-300 underline-offset-4 cursor-pointer transition-colors px-1 py-0.5 rounded hover:bg-sky-500/10"><svg class="w-3 h-3 shrink-0 fill-current text-sky-400" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg><span>${safeLabel}</span></a>`
      );
      return `${citeTokenPrefix}${idx}ZTOKEN`;
    }
  );

  // 3b. Extract clickable slide/material citations [[slide:SOURCE_ID#PAGE|LABEL]] or [[ref:...]] or [[doc:...]]
  text = text.replace(
    /\[\[(?:ref|slide|doc):([^\]|#]+?)(?:[#|](\d+))?\|([^\]]+?)\]\]/g,
    (_match, sourceId: string, page: string | undefined, label: string) => {
      const idx = citationTokens.length;
      const safeSource = escapeHtml(sourceId.trim());
      const safeLabel = escapeHtml(label.trim());
      const pageAttr = page ? ` data-slide-page="${escapeHtml(page)}"` : '';
      citationTokens.push(
        `<a href="#slide-ref" data-slide-source="${safeSource}"${pageAttr} class="slide-citation-link inline-flex items-center gap-1 font-medium text-emerald-300 hover:text-emerald-100 underline decoration-emerald-400/60 hover:decoration-emerald-300 underline-offset-4 cursor-pointer transition-colors px-1 py-0.5 rounded hover:bg-emerald-500/10"><svg class="w-3 h-3 shrink-0 stroke-current text-emerald-400 fill-none" viewBox="0 0 24 24" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg><span>${safeLabel}</span></a>`
      );
      return `${citeTokenPrefix}${idx}ZTOKEN`;
    }
  );

  // 3c. Extract clickable internal settings link [[action:settings|LABEL]]
  text = text.replace(/\[\[action:settings\|([^\]]+?)\]\]/g, (_match, label: string) => {
    const idx = citationTokens.length;
    const safeLabel = escapeHtml(label.trim());
    citationTokens.push(
      `<a href="#open-settings" data-open-settings="true" class="underline decoration-white/55 hover:decoration-white underline-offset-4 text-white font-medium cursor-pointer transition-colors">${safeLabel}</a>`
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
      'data-audio-lecture-id',
      'data-audio-seconds',
      'data-open-settings',
      'data-code-copy',
      'data-code-index',
      'target',
      'rel',
      'aria-hidden',
      'viewBox',
      'strokeWidth',
      'stroke-width',
      'stroke-linecap',
      'stroke-linejoin',
      'fill',
      'stroke',
      'd',
    ],
    ADD_TAGS: ['svg', 'path'],
  });

  // 8. Restore sanitized KaTeX HTML (nonce prefix prevents collisions with literal text in code blocks)
  const katexTokenRegex = new RegExp(`${katexTokenPrefix}(\\d+)ZTOKEN`, 'g');
  html = html.replace(katexTokenRegex, (_match, indexStr: string) => {
    return katexTokens[Number(indexStr)] ?? '';
  });

  return html;
}

const MarkdownContent = memo(function MarkdownContent({
  content,
  className = '',
  onOpenCitation,
  onOpenAudioSnippet,
  onOpenSettings,
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

    // 0a. Clickable audio playback citation link
    const audioEl = target?.closest('a[data-audio-lecture-id], button[data-audio-lecture-id]') as HTMLElement | null;
    if (audioEl) {
      e.preventDefault();
      const lectureIdStr = audioEl.getAttribute('data-audio-lecture-id');
      const secondsStr = audioEl.getAttribute('data-audio-seconds');
      if (lectureIdStr) {
        const lectureId = Number.parseInt(lectureIdStr, 10);
        const seconds = secondsStr ? Math.max(0, Number.parseFloat(secondsStr)) : 0;
        if (Number.isFinite(lectureId)) {
          onOpenAudioSnippet?.(lectureId, Number.isFinite(seconds) ? seconds : 0, true);
        }
      }
      return;
    }

    // 0b. Internal settings action link
    const settingsLink = target?.closest('a[data-open-settings], button[data-open-settings]') as HTMLElement | null;
    if (settingsLink) {
      e.preventDefault();
      onOpenSettings?.();
      return;
    }

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
});

export default MarkdownContent;
