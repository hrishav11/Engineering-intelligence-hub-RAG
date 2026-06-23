"""Capture dark-mode UI screenshots for the README.

Drives the running Streamlit app at http://localhost:8501 with Playwright:
1. Toggles to Dark mode
2. Screenshot of the empty/hero state
3. Asks an example question, screenshots the answer + sources
4. Asks a vague question, screenshots the agentic fallback banner

Outputs to docs/screenshots/.
"""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)


def wait_for_app(page):
    page.wait_for_selector("text=Engineering Intelligence", timeout=30_000)
    page.wait_for_timeout(800)


def set_dark_mode(page):
    try:
        toggle = page.locator('[data-testid="stSidebarCollapseButton"]')
        if toggle.is_visible(timeout=1000):
            toggle.click()
            page.wait_for_timeout(300)
    except Exception:
        pass
    page.locator('label:has-text("Dark")').first.click()
    page.wait_for_timeout(600)


def ask(page, question: str):
    textarea = page.locator('textarea').first
    textarea.click()
    textarea.press_sequentially(question, delay=10)
    # Streamlit's text_area doesn't sync to the server until focus leaves OR
    # Ctrl+Enter is pressed. Ctrl+Enter triggers the rerun, which updates session_state
    # and enables the button.
    page.keyboard.press("Control+Enter")
    page.wait_for_timeout(2000)
    # The Ctrl+Enter rerun may also click the form-default button, but we have no
    # form, so just click the Ask button now.
    btn = page.locator('[data-testid="stBaseButton-primary"]').first
    page.wait_for_function(
        "() => { const b = document.querySelector('[data-testid=\\\"stBaseButton-primary\\\"]'); return b && !b.disabled; }",
        timeout=15_000,
    )
    btn.click()


def wait_for_answer(page, timeout_ms: int = 120_000):
    page.wait_for_selector('.eih-label:has-text("Answer")', timeout=timeout_ms)
    page.wait_for_timeout(1500)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        page = ctx.new_page()

        page.goto(URL)
        wait_for_app(page)
        set_dark_mode(page)
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT / "01_hero_dark.png"), full_page=False)
        print("OK 01_hero_dark.png")

        ask(page, "How does BaseXCom.serialize_value work?")
        wait_for_answer(page)
        page.screenshot(path=str(OUT / "02_answer_hybrid_dark.png"), full_page=True)
        print("OK 02_answer_hybrid_dark.png")

        page.goto(URL)
        wait_for_app(page)
        set_dark_mode(page)
        page.wait_for_timeout(500)
        ask(page, "How does BashOperator execute its command?")
        wait_for_answer(page, timeout_ms=180_000)
        page.screenshot(path=str(OUT / "03_answer_agentic_fallback_dark.png"), full_page=True)
        print("OK 03_answer_agentic_fallback_dark.png")

        ctx.close()
        browser.close()
        print(f"\nSaved to: {OUT}")


if __name__ == "__main__":
    main()
