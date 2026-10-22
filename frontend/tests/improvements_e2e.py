"""Browser regression checks for experiments, replay sandbox and persisted pause.

Run with both servers running: python frontend/tests/improvements_e2e.py
Use CARO_DATABASE_PATH and CARO_BENCHMARK_PATH on the backend for isolated runs.
"""
import json
from pathlib import Path
import re
from playwright.sync_api import sync_playwright,expect

OUT = Path(__file__).resolve().parents[1]/"test-results"
OUT.mkdir(exist_ok=True)


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width":1440,"height":1100},base_url="http://127.0.0.1:5173",accept_downloads=True)
        errors = []
        page = context.new_page(); page.on("pageerror",lambda error: errors.append(str(error)))
        data = context.request.post("/api/games",data={"size":3,"mode":"local"}).json()
        game_id,token = data["game"]["id"],data["token"]
        for revision,(r,c) in enumerate([(0,0),(1,0),(0,1),(1,1),(2,2),(1,2)]):
            response = context.request.post(f"/api/games/{game_id}/move",headers={"X-Player-Token":token},data={"row":r,"col":c,"revision":revision})
            assert response.ok,response.text()
        page.add_init_script(f"if (!localStorage.getItem('caro.ids')) localStorage.setItem('caro.ids', JSON.stringify([{json.dumps(game_id)}]));")
        page.goto("/")
        page.get_by_role("button",name=re.compile("AI Lab")).first.click()
        expect(page.get_by_role("heading",name="Đánh giá trên tập test giữ riêng")).to_be_visible()
        accuracy = context.request.get("/api/model").json()["training"]["test_metrics"]["top1_accuracy"]
        expect(page.locator(".evaluation-panel .evaluation-values")).to_contain_text(f"{accuracy*100:.1f}%")
        expect(page.locator(".benchmark-table tbody tr")).to_have_count(3)
        with page.expect_download() as download:
            page.get_by_role("button",name="Tải kết quả JSON").click()
        path = OUT/"benchmark-download.json"; download.value.save_as(path)
        report = json.loads(path.read_text(encoding="utf-8")); assert "matches" in report and report["total_games"] > 0
        page.get_by_label("Bàn thí nghiệm").select_option("3")
        page.get_by_label("Ngân sách tìm kiếm").select_option("40")
        page.get_by_label("So sánh cả ba engine").uncheck()
        page.get_by_label("Engine A").select_option("cnn")
        page.get_by_label("Engine B").select_option("heuristic")
        page.get_by_role("button",name="Chạy thí nghiệm",exact=True).click()
        expect(page.get_by_role("heading",name="Kết quả 2 ván",exact=True)).to_be_visible(timeout=20000)
        expect(page.locator(".benchmark-table tbody tr")).to_have_count(2)
        page.locator(".experiment-panel").screenshot(path=str(OUT/"experiments.png"))
        assert context.request.post("/api/experiments",data={"variants":["cnn","cnn"],"sizes":[100]}).status==422
        print("PASS held-out metrics, JSON export, real background experiment and input validation",flush=True)

        page.get_by_role("button",name="Thống kê",exact=True).click()
        expect(page.locator("tbody tr")).to_have_count(1)
        page.get_by_role("button",name="Xem lại",exact=False).click()
        dialog = page.get_by_role("dialog")
        dialog.get_by_role("button",name="Phân tích cơ hội sau ván").click()
        expect(dialog.locator(".review-summary")).to_contain_text("X: bỏ lỡ 1 nước thắng")
        dialog.get_by_role("button",name="Trước nước 5 · X",exact=True).click()
        expect(dialog.locator(".occupied")).to_have_count(4)
        dialog.get_by_role("button",name="Thử phương án khác").click()
        dialog.get_by_role("button",name="Thử C1",exact=True).click()
        expect(dialog.locator(".practice-note")).to_contain_text("X thắng trong phương án thử")
        expect(dialog.locator(".occupied")).to_have_count(5)
        dialog.screenshot(path=str(OUT/"review-sandbox.png"))
        stored = context.request.post("/api/history",data={"ids":[game_id]}).json()[0]
        assert len(stored["moves"])==6 and stored["winner"]==-1
        dialog.get_by_role("button",name="Trở về ván gốc").click()
        expect(dialog.locator(".occupied")).to_have_count(4)
        dialog.get_by_role("button",name="Đóng xem lại",exact=True).click()
        print("PASS tactical review, alternate winning move and immutable original history",flush=True)

        page.get_by_role("button",name="Chơi cờ",exact=True).click()
        page.locator(".mode-option").filter(has_text="AI đấu AI").click()
        page.get_by_role("button",name="3×3",exact=True).click()
        page.get_by_role("button",name="Tùy chỉnh nâng cao").click()
        page.get_by_label("Thời gian mỗi lượt").select_option("15")
        page.get_by_role("button",name="Lưu thiết lập",exact=False).click()
        page.locator(".start-button").click()
        expect(page.locator(".occupied").first).to_be_visible(timeout=10000)
        page.get_by_role("button",name="Tạm dừng AI",exact=True).click()
        expect(page.locator(".turn-banner")).to_contain_text("tạm dừng")
        remaining,count = page.locator(".timer").inner_text(),page.locator(".occupied").count()
        page.wait_for_timeout(1600)
        assert page.locator(".timer").inner_text()==remaining
        assert page.locator(".occupied").count()==count
        page.reload()
        expect(page.locator(".ready-overlay")).to_be_visible()
        page.get_by_role("button",name="Tiếp tục ván đã lưu",exact=True).click()
        expect(page.locator(".turn-banner")).to_contain_text("tạm dừng")
        expect(page.locator(".occupied")).to_have_count(count)
        assert page.locator(".timer").inner_text()==remaining
        page.get_by_role("button",name="Tiếp tục AI",exact=True).click()
        expect(page.locator(".occupied").nth(count)).to_be_visible(timeout=10000)
        print("PASS frozen clock, explicit reload resume, and continued AI moves",flush=True)

        page.locator(".mode-option").filter(has_text="Đấu với AI").click()
        page.get_by_role("button",name="5×5",exact=True).click()
        page.get_by_role("button",name="Tùy chỉnh nâng cao").click()
        page.get_by_label("Engine AI").select_option("cnn")
        page.get_by_label("Thời gian mỗi lượt").select_option("0")
        page.get_by_role("button",name="Lưu thiết lập",exact=False).click()
        page.locator(".start-button").click()
        page.get_by_role("gridcell",name="C3: ô trống",exact=True).click()
        expect(page.locator(".occupied")).to_have_count(2,timeout=15000)
        expect(page.locator(".analysis-panel")).to_contain_text("Chọn trực tiếp từ policy CNN")
        page.locator(".analysis-panel summary").click()
        expect(page.locator(".decision-details")).to_contain_text("Trọng số CNN: 100%")
        expect(page.locator(".selected-candidate")).to_have_count(1)
        page.locator(".analysis-panel").screenshot(path=str(OUT/"decision-details.png"))
        page.get_by_role("gridcell",name="A1: ô trống",exact=True).click()
        expect(page.locator(".occupied")).to_have_count(4,timeout=15000)
        print("PASS engine selection, component scores, and continued legal play",flush=True)

        mobile = browser.new_context(viewport={"width":390,"height":844},is_mobile=True,has_touch=True)
        mobile_page = mobile.new_page(); mobile_page.on("pageerror",lambda error: errors.append(str(error)))
        mobile_page.goto("http://127.0.0.1:5173/")
        mobile_page.get_by_role("button",name=re.compile("AI Lab")).first.click()
        expect(mobile_page.locator(".experiment-panel")).to_be_visible()
        assert mobile_page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"AI Lab overflows on mobile"
        mobile_page.screenshot(path=str(OUT/"mobile-lab.png"),full_page=True)
        assert not errors,json.dumps(errors,ensure_ascii=True)
        print("ALL IMPROVEMENT BROWSER CHECKS PASSED; no JavaScript errors",flush=True)
        browser.close()


if __name__ == "__main__": main()
