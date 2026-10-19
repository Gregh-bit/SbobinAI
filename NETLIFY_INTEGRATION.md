# Bottone di download per il sito Netlify

Incolla questo blocco dove vuoi il pulsante. Rileva macOS o Windows e scarica l'ultima release da GitHub con un clic;
sotto mostra il link per l'altro sistema.

```html
<div id="sbobinai-download">
  <a id="sbobinai-main" class="sb-btn" href="https://github.com/Gregh-bit/SbobinAI/releases/latest">Scarica SbobinAI</a>
  <p class="sb-alt">Altro sistema: <a id="sbobinai-alt" href="https://github.com/Gregh-bit/SbobinAI/releases/latest">tutte le versioni</a></p>
</div>

<style>
  #sbobinai-download { text-align: center; font-family: system-ui, sans-serif; }
  .sb-btn { display: inline-block; padding: 14px 28px; border-radius: 14px; background: #6366f1; color: #fff;
            font-weight: 600; text-decoration: none; box-shadow: 0 6px 20px rgba(99,102,241,.35); transition: transform .15s; }
  .sb-btn:hover { transform: translateY(-2px); }
  .sb-alt { margin-top: 10px; font-size: 14px; opacity: .75; }
</style>

<script>
(async () => {
  const REPO = "Gregh-bit/SbobinAI";
  const FALLBACK = "https://github.com/" + REPO + "/releases/latest";
  const platform = (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || "";
  const isMac = /mac/i.test(platform) || /Macintosh/.test(navigator.userAgent);
  const main = document.getElementById("sbobinai-main");
  const alt = document.getElementById("sbobinai-alt");
  const find = (assets, re) => (assets.find(a => re.test(a.name)) || {}).browser_download_url;
  try {
    const res = await fetch("https://api.github.com/repos/" + REPO + "/releases/latest");
    if (!res.ok) throw new Error(res.status);
    const rel = await res.json();
    const mac = { label: "macOS (.dmg)", url: find(rel.assets, /\.dmg$/i) };
    const win = { label: "Windows (.exe)", url: find(rel.assets, /setup.*\.exe$/i) || find(rel.assets, /\.exe$/i) || find(rel.assets, /\.msi$/i) };
    const primary = isMac ? mac : win;
    const other = isMac ? win : mac;
    main.textContent = "Scarica per " + primary.label + " — " + rel.tag_name;
    main.href = primary.url || FALLBACK;
    alt.textContent = "Scarica per " + other.label;
    alt.href = other.url || FALLBACK;
  } catch (e) {
    main.href = FALLBACK; // GitHub non raggiungibile: apre la pagina delle release
  }
})();
</script>
```

Note
- Per Windows preferisce l'installer NSIS (`...setup.exe`), poi qualunque `.exe`, poi `.msi`.
- L'API pubblica di GitHub ha un limite di 60 richieste/ora per IP: in caso di errore il pulsante punta alla pagina delle release.
- Su Linux o altri sistemi il pulsante principale propone Windows e il link secondario macOS.
