/*
 * Copy-as-Markdown actions for each documentation page.
 *
 * The llmstxt plugin writes a .md counterpart next to every rendered page, so
 * the source is just the current path with the extension swapped. The controls
 * start hidden and are revealed only once that file is confirmed to exist,
 * which keeps them off any page the plugin does not cover.
 */

function markdownUrlForCurrentPage() {
  const path = window.location.pathname;
  // use_directory_urls is false, so pages are served as *.html. Directory-style
  // URLs are still possible for the site root.
  if (path.endsWith(".html")) {
    return path.replace(/\.html$/, ".md");
  }
  return path.replace(/\/$/, "") + "/index.md";
}

function initPageActions() {
  const container = document.querySelector("[data-ik-page-actions]");
  if (!container) {
    return;
  }

  const button = container.querySelector("[data-ik-copy]");
  const label = container.querySelector("[data-ik-copy-label]");
  const view = container.querySelector("[data-ik-view]");
  const defaultLabel = label.textContent;
  const url = markdownUrlForCurrentPage();

  view.href = url;

  // A HEAD request avoids pulling the whole file just to decide whether to
  // show the controls.
  fetch(url, { method: "HEAD" })
    .then((response) => {
      if (response.ok) {
        container.hidden = false;
      }
    })
    .catch(() => {
      /* Leave the controls hidden. */
    });

  let resetTimer = null;

  button.addEventListener("click", async () => {
    if (resetTimer) {
      window.clearTimeout(resetTimer);
    }
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Request failed with ${response.status}`);
      }
      // Requires a secure context; GitHub Pages and localhost both qualify.
      await navigator.clipboard.writeText(await response.text());
      label.textContent = "Copied";
    } catch (error) {
      label.textContent = "Copy failed";
    }
    resetTimer = window.setTimeout(() => {
      label.textContent = defaultLabel;
    }, 2000);
  });
}

// document$ is Material's per-page observable, which also fires after instant
// navigation. Fall back to a plain listener if the theme bundle is absent.
if (typeof document$ !== "undefined") {
  document$.subscribe(initPageActions);
} else {
  document.addEventListener("DOMContentLoaded", initPageActions);
}
