"""The app in a real browser: clicks, hover, lookups, URL state and narrow screens.

Needs Playwright and an installed Chrome (`pip install -r requirements-dev.txt`); skipped otherwise.
Starts its own server on port 8051.
"""

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).parent.parent
PORT = 8051
BASE = f"http://127.0.0.1:{PORT}"


@pytest.fixture(scope="session")
def server():
    env = dict(os.environ, PORT=str(PORT), DASH_DEBUG="0")
    proc = subprocess.Popen([sys.executable, "app.py"], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for _ in range(120):
        try:
            urllib.request.urlopen(BASE, timeout=1)
            break
        except OSError:
            if proc.poll() is not None:
                pytest.fail(proc.stderr.read().decode())
            time.sleep(0.5)
    yield BASE
    proc.terminate()
    proc.wait(10)


@pytest.fixture(scope="session")
def browser():
    with playwright.sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="chrome")
        except Exception as e:  # no Chrome installed
            pytest.skip(f"Chrome not available: {e}")
        yield b
        b.close()


@pytest.fixture
def page(browser, server):
    context = browser.new_context(viewport={"width": 1200, "height": 900})
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    yield page
    context.close()
    assert not errors, errors


def wait_for_text(locator, predicate, timeout=60):
    """Wait until the locator's text satisfies `predicate`; return the text."""
    deadline = time.time() + timeout
    text = ""
    while time.time() < deadline:
        text = locator.inner_text()
        if predicate(text):
            return text
        time.sleep(0.25)
    raise AssertionError(f"text never matched: {text!r}")


def centre(locator):
    """Centre of an element on screen. Plotly handles the mouse on a layer above its marks, so
    tests move a real mouse there instead of dispatching events to the marks themselves."""
    locator.scroll_into_view_if_needed()
    box = locator.bounding_box()
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


def loaded(page, graph_id, timeout=90_000):
    page.wait_for_selector(f"#{graph_id} .main-svg", timeout=timeout)
    page.wait_for_function(f"!document.querySelector('#{graph_id}').closest('[data-dash-is-loading]')",
                           timeout=timeout)


@pytest.mark.parametrize("path, graphs", [
    ("/", ["chart"]),
    ("/vocabulary", ["vocab-cloud", "vocab-detail"]),
    ("/countries", ["geo-map", "geo-detail"]),
    ("/ai", ["ai-trend", "ai-heatmap", "ai-style"]),
])
def test_pages_render(page, path, graphs):
    page.goto(BASE + path)
    for graph in graphs:
        loaded(page, graph)
        assert page.locator(f"#{graph} .main-svg").count() > 0


def test_cloud_click_shows_word_history(page):
    page.goto(BASE + "/vocabulary")
    loaded(page, "vocab-cloud")
    title = page.locator("#vocab-detail-title")
    first = wait_for_text(title, lambda t: "over time" in t)
    words = page.locator("#vocab-cloud g.textpoint text")
    assert words.count() >= 50
    target = words.nth(words.count() - 1)  # a small word, not the default one
    label = target.text_content()
    page.mouse.click(*centre(target))
    wait_for_text(title, lambda t: label in t)
    assert first != title.inner_text()
    assert "word=" in page.url


def test_cloud_hover_shows_values(page):
    page.goto(BASE + "/vocabulary")
    loaded(page, "vocab-cloud")
    word = page.locator("#vocab-cloud g.textpoint text").first
    page.mouse.move(*centre(word))
    page.wait_for_selector("#vocab-cloud .hoverlayer .hovertext", timeout=10_000)
    assert "expected" in page.locator("#vocab-cloud .hoverlayer").text_content()


def test_word_lookup(page):
    page.goto(BASE + "/vocabulary")
    loaded(page, "vocab-cloud")
    page.fill("#vocab-search", "Bohužel")
    page.press("#vocab-search", "Enter")
    wait_for_text(page.locator("#vocab-detail-title"), lambda t: "bohužel" in t)
    assert "Rank" in wait_for_text(page.locator("#vocab-summary"), lambda t: "Rank" in t)
    page.fill("#vocab-search", "xyzzyq")
    page.press("#vocab-search", "Enter")
    wait_for_text(page.locator("#vocab-detail-title"), lambda t: "not found" in t)


def test_map_click_selects_country(page):
    page.goto(BASE + "/countries")
    loaded(page, "geo-map")
    title = page.locator("#geo-detail-title")
    first = wait_for_text(title, lambda t: "over time" in t)
    # Click Ukraine: find its shape through the data Plotly attaches to each path. The map renders
    # again once the theme is known, so wait for the shape to be there.
    find = """Array.from(document.querySelectorAll('#geo-map path.choroplethlocation'))
        .find(el => el.__data__ && el.__data__.loc === 'UKR')"""
    page.wait_for_function(f"() => !!({find})", timeout=30_000)
    page.wait_for_timeout(500)
    box = page.evaluate("""() => {
        const p = Array.from(document.querySelectorAll('#geo-map path.choroplethlocation'))
            .find(el => el.__data__ && el.__data__.loc === 'UKR');
        const r = p.getBoundingClientRect();
        return {x: r.x + r.width / 2, y: r.y + r.height / 2};
    }""")
    page.mouse.click(box["x"], box["y"])
    wait_for_text(title, lambda t: "Ukraine" in t)
    assert "Ukraine" in title.inner_text() and first != title.inner_text()
    assert "country=UKR" in page.url


def test_settings_live_in_the_url(page):
    page.goto(BASE + "/countries")
    loaded(page, "geo-map")
    page.get_by_label("Average of the baseline years").check()
    page.wait_for_function("location.search.includes('baseline=mean')", timeout=10_000)
    # A shared link restores the settings.
    page.goto(BASE + "/countries?baseline=mean&lang=en")
    loaded(page, "geo-map")
    assert page.get_by_label("Average of the baseline years").is_checked()
    assert page.get_by_label("English").is_checked()
    # The back button steps back through earlier settings.
    page.get_by_label("Czech").check()
    page.wait_for_function("!location.search.includes('lang=en')", timeout=10_000)
    page.go_back()
    page.wait_for_function("location.search.includes('lang=en')", timeout=10_000)
    page.wait_for_function("document.querySelector('input[value=en]') && document.querySelector('input[value=en]').checked",
                           timeout=30_000)


def test_compare_faculties_view(page):
    page.goto(BASE + "/?view=compare")
    loaded(page, "chart")
    page.wait_for_function("document.querySelectorAll('#chart .scatterlayer .trace').length >= 8", timeout=30_000)


def test_phone_width_has_no_sideways_scroll(browser, server):
    context = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True)
    page = context.new_page()
    page.goto(BASE + "/vocabulary")
    loaded(page, "vocab-cloud")
    width = page.evaluate("document.querySelector('#vocab-cloud .main-svg').getBoundingClientRect().width")
    assert width <= 390
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
    context.close()
