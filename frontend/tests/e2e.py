"""Optional browser integration checks.

With both servers running: python frontend/tests/e2e.py
Requires: pip install playwright && python -m playwright install chromium
"""
from pathlib import Path
import json
import re
from playwright.sync_api import sync_playwright, expect

OUT = Path(__file__).resolve().parents[1] / "test-results"
OUT.mkdir(exist_ok=True)


def main():
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width":1440,"height":1100}, device_scale_factor=1)
        page = context.new_page()
        page.on("pageerror",lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:5173")
        expect(page.get_by_role("grid")).to_be_visible()
        expect(page.locator(".cell")).to_have_count(225)
        page.screenshot(path=str(OUT/"desktop.png"),full_page=True)
        page.get_by_role("gridcell",name="H8: ô trống",exact=True).click()
        expect(page.locator(".occupied")).to_have_count(2,timeout=15000)
        page.get_by_role("button",name="Gợi ý",exact=True).click()
        expect(page.locator(".hint-cell")).to_have_count(1)
        page.get_by_role("button",name="Đi lại",exact=True).click()
        expect(page.locator(".occupied")).to_have_count(0)
        print("PASS AI turn, neural hint, paired undo",flush=True)

        page.locator(".mode-option").filter(has_text="Hai người").click()
        page.get_by_role("button",name="3×3",exact=True).click()
        page.locator(".start-button").click()
        expect(page.locator(".cell")).to_have_count(9)
        for coord in ["A1","B1","A2","B2","A3"]:
            page.get_by_role("gridcell",name=f"{coord}: ô trống",exact=True).click()
            expect(page.get_by_role("gridcell",name=f"{coord}: ô trống",exact=True)).to_have_count(0)
        expect(page.locator(".winning")).to_have_count(3)
        expect(page.locator(".finished-banner")).to_be_visible()
        page.screenshot(path=str(OUT/"local-win.png"),full_page=True)
        page.get_by_role("button",name="Thống kê",exact=True).click()
        expect(page.locator("tbody tr")).to_have_count(1)
        page.get_by_role("button",name="Xem lại",exact=False).click()
        expect(page.locator("dialog .occupied")).to_have_count(5)
        page.get_by_role("slider",name="Chọn nước đi").fill("0")
        expect(page.locator("dialog .occupied")).to_have_count(0)
        page.get_by_role("button",name="Đóng xem lại",exact=True).click()
        print("PASS local win, durable history, replay",flush=True)

        page.get_by_role("button",name="Hướng dẫn",exact=True).click()
        expect(page.locator(".guide-card")).to_have_count(6)
        page.get_by_role("button",name=re.compile("AI Lab")).first.click()
        expect(page.locator(".training-chart svg")).to_be_visible()
        expect(page.get_by_text("33.602",exact=True)).to_be_visible()
        page.screenshot(path=str(OUT/"ai-lab.png"),full_page=True)
        page.get_by_role("button",name="Chuyển giao diện tối").click()
        expect(page.locator("html")).to_have_attribute("data-theme","dark")
        page.get_by_role("button",name="Chơi cờ",exact=True).click()
        page.screenshot(path=str(OUT/"dark.png"),full_page=True)
        page.get_by_role("button",name="Chuyển giao diện sáng").click()
        print("PASS guide, actual training chart, dark mode",flush=True)

        page.locator(".mode-option").filter(has_text="Hai người").click()
        page.get_by_role("button",name="Tùy chỉnh nâng cao").click()
        page.get_by_label("Kích thước tùy chỉnh").fill("25")
        page.get_by_label("Số quân để thắng").select_option("6")
        page.get_by_label("Luật thắng").select_option("blocked")
        page.get_by_role("button",name="Lưu thiết lập",exact=False).click()
        page.locator(".start-button").click()
        expect(page.locator(".cell")).to_have_count(625)
        expect(page.locator(".board-caption")).to_contain_text("6 quân thắng")
        page.get_by_role("gridcell",name="Y25: ô trống",exact=True).click()
        expect(page.get_by_role("gridcell",name="Y25: X",exact=True)).to_be_visible()
        print("PASS custom 25x25, six-in-row, blocked rule, corner move",flush=True)

        page.get_by_role("button",name="3×3",exact=True).click()
        page.locator(".mode-option").filter(has_text="AI đấu AI").click()
        page.locator(".start-button").click()
        expect(page.locator(".occupied").first).to_be_visible(timeout=10000)
        page.get_by_role("button",name="Tạm dừng AI").click()
        count=page.locator(".occupied").count()
        page.wait_for_timeout(1200)
        assert page.locator(".occupied").count()==count
        page.get_by_role("button",name="Tiếp tục AI").click()
        expect(page.locator(".occupied").nth(count)).to_be_visible(timeout=10000)
        print("PASS AI exhibition pause/resume",flush=True)

        page.locator(".mode-option").filter(has_text="Chơi online").click()
        page.locator(".start-button").click()
        expect(page.locator(".room-banner strong")).to_be_visible()
        code=page.locator(".room-banner strong").inner_text()
        guest_context=browser.new_context(viewport={"width":1280,"height":900})
        guest=guest_context.new_page()
        guest.on("pageerror",lambda error: errors.append(str(error)))
        guest.goto("http://127.0.0.1:5173")
        expect(guest.get_by_role("grid")).to_be_visible()
        guest.locator(".mode-option").filter(has_text="Chơi online").click()
        guest.get_by_role("button",name="Tham gia bằng mã phòng").click()
        guest.get_by_label("Tên người chơi",exact=True).fill("Người bạn")
        guest.get_by_label("Mã phòng",exact=True).fill(code)
        guest.get_by_role("button",name="Tham gia phòng",exact=True).click()
        expect(page.locator(".turn-banner")).to_contain_text("Đến lượt")
        page.get_by_role("gridcell",name="A1: ô trống",exact=True).click()
        expect(guest.get_by_role("gridcell",name="A1: X",exact=True)).to_be_visible()
        guest.get_by_role("gridcell",name="B1: ô trống",exact=True).click()
        expect(page.get_by_role("gridcell",name="B1: O",exact=True)).to_be_visible()
        guest.reload()
        expect(guest.get_by_role("gridcell",name="B1: O",exact=True)).to_be_visible()
        page.screenshot(path=str(OUT/"online.png"),full_page=True)
        print("PASS two-browser online room, websocket sync, reload recovery",flush=True)

        mobile_context=browser.new_context(viewport={"width":390,"height":844},is_mobile=True,has_touch=True,device_scale_factor=1)
        mobile=mobile_context.new_page()
        mobile.on("pageerror",lambda error: errors.append(str(error)))
        mobile.goto("http://127.0.0.1:5173")
        expect(mobile.get_by_role("grid")).to_be_visible()
        assert mobile.evaluate("document.documentElement.scrollWidth <= window.innerWidth"),"Mobile has horizontal overflow"
        mobile.get_by_role("gridcell",name="H8: ô trống",exact=True).click()
        expect(mobile.locator(".occupied")).to_have_count(2,timeout=15000)
        mobile.screenshot(path=str(OUT/"mobile.png"),full_page=True)
        print("PASS mobile layout and playable AI board",flush=True)
        assert not errors, json.dumps(errors,ensure_ascii=True)
        print("ALL BROWSER CHECKS PASSED; no JavaScript errors",flush=True)
        browser.close()


if __name__=="__main__":
    main()
