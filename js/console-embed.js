/**
 * OPTIONAL live embed of the hosted Python console.
 *
 * Empty is the SHIPPED state, and it is not a missing feature: Hugging Face moved
 * Docker and Gradio Spaces on free CPU behind a PRO subscription, and only static
 * Spaces remain free, so the Streamlit console in hf_space/ has no free host. The
 * console section on technical.html renders its own content -- what that console adds
 * over the browser page, and how to run it locally -- straight from the HTML, with no
 * placeholder and nothing deferred.
 *
 * Paste a URL below (e.g. 'https://<owner>-<space>.hf.space') and wireConsole()
 * replaces that content with the live embed. Nothing else needs to change. This lives
 * in its own module so that there is exactly ONE constant to edit no matter which
 * page carries the console section: both js/app.js and js/tech.js call wireConsole(),
 * and it is a no-op on a page with no #console-slot.
 */
export const HF_SPACE_URL = '';

/**
 * Upgrades the console section to a live embed IF a host URL exists.
 *
 * The default path returns immediately and leaves the page's own content standing:
 * what the Python console adds over this page, and how to run it. That content is the
 * shipped state and it is complete -- there is no placeholder to fill, and nothing on
 * the page is waiting on this function. Which is why this reads the static block
 * rather than writing one: with JavaScript disabled the section still says everything
 * it needs to.
 */
export function wireConsole() {
  if (!HF_SPACE_URL) return;

  const slot = document.getElementById('console-slot');
  const stat = document.getElementById('console-static');
  if (!slot) return;
  if (stat) stat.remove();

  const frame = document.createElement('iframe');
  frame.src = HF_SPACE_URL;
  frame.title = 'Jindal Stainless surface inspection console';
  frame.loading = 'lazy';
  frame.allow = 'clipboard-write; fullscreen';
  slot.appendChild(frame);

  const a = document.getElementById('console-link');
  if (a) { a.href = HF_SPACE_URL; a.hidden = false; }
}

