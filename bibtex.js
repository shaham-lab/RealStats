document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("copy-bibtex-btn");
    const notif = document.getElementById("bibtex-notification");
    const bibtexEl = document.querySelector(".citation-box");

    if (!btn || !notif || !bibtexEl) return;

    btn.addEventListener("click", (e) => {
        e.preventDefault();

        // Read raw innerText (this preserves newlines)
        let bibtex = bibtexEl.innerText;

        // Remove HTML artifacts (like leading/trailing spaces from &nbsp;)
        bibtex = bibtex.replace(/\u00a0/g, ""); // non-breaking space
        bibtex = bibtex.replace(/^\s+/gm, "");  // indentation cleanup

        navigator.clipboard.writeText(bibtex).then(() => {
            const rect = btn.getBoundingClientRect();
            notif.style.left = rect.left + rect.width / 2 + "px";
            notif.style.top = rect.bottom + 8 + window.scrollY + "px";

            notif.classList.add("show");
            setTimeout(() => notif.classList.remove("show"), 1500);
        });
    });
});