"""End-to-end human-history training. Run only against an isolated QA database/model.

Both servers must be running. This creates test games and may promote a QA model.
python frontend/tests/human_learning_e2e.py
"""
import json
from pathlib import Path
import random
import re
import time
from playwright.sync_api import sync_playwright,expect

OUT=Path(__file__).resolve().parents[1]/"test-results"
OUT.mkdir(exist_ok=True)


def create_finished_game(api,rng,size=5,k=4):
    data=api.post("/api/games",data={"mode":"local","size":size,"win_length":k}).json()
    headers={"X-Player-Token":data["token"]}
    empty=list(range(size*size)); rng.shuffle(empty)
    for revision,index in enumerate(empty):
        r,c=divmod(index,size)
        response=api.post(f'/api/games/{data["game"]["id"]}/move',headers=headers,data={"row":r,"col":c,"revision":revision})
        assert response.ok,response.text()
        if response.json()["status"] in ("won","draw"): return
    raise AssertionError("Expected a terminal game")


def main():
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(headless=True)
        context=browser.new_context(viewport={"width":1440,"height":1100},base_url="http://127.0.0.1:5173",accept_downloads=True)
        errors=[]
        page=context.new_page(); page.on("pageerror",lambda e:errors.append(str(e)))
        response=context.request.post("/api/learning/settings",data={"enabled":False,"epochs":1,"min_new_games":3})
        assert response.ok,response.text()
        rng=random.Random(2026+context.request.get("/api/learning").json()["stats"]["total_games"])
        for _ in range(10): create_finished_game(context.request,rng)
        stats=context.request.get("/api/learning").json()["stats"]
        assert stats["pending_games"]>=8 and stats["train_games"]>=2
        parent=context.request.get("/api/model").json()["checkpoint_sha256"]
        page.goto("/")
        page.get_by_role("button",name=re.compile("AI Lab")).first.click()
        panel=page.locator(".human-learning-panel")
        expect(panel.get_by_role("heading",name="Học từ các ván với người chơi",exact=True)).to_be_visible()
        expect(panel.locator(".learning-stats strong").first).to_have_text(str(stats["total_games"]))
        expect(panel.get_by_label("Tự học khi đủ ván mới")).not_to_be_checked()
        panel.get_by_label("Epoch mỗi lần học").fill("1")
        panel.get_by_role("button",name="Lưu thiết lập học",exact=True).click()
        expect(panel.get_by_role("button",name="Học từ lịch sử ngay",exact=True)).to_be_enabled()
        prior=context.request.get("/api/learning").json().get("report")
        prior_id=prior["run_id"] if prior else None
        panel.get_by_role("button",name="Học từ lịch sử ngay",exact=True).click()
        expect(panel.get_by_role("button",name="Đang học và đánh giá…",exact=True)).to_be_disabled()
        assert context.request.get("/api/health").json()["model_loaded"]
        page.get_by_role("button",name="Chơi cờ",exact=True).click()
        page.locator(".start-button").click()
        page.get_by_role("gridcell",name="H8: ô trống",exact=True).click()
        expect(page.locator(".occupied")).to_have_count(2,timeout=15000)
        page.get_by_role("button",name=re.compile("AI Lab")).first.click()
        panel=page.locator(".human-learning-panel")
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            state=context.request.get("/api/learning").json()
            if state["status"]!="running" and state.get("report") and state["report"]["run_id"]!=prior_id: break
            page.wait_for_timeout(500)
        report=state["report"]
        assert state["status"]=="completed",json.dumps({k:state.get(k) for k in ("status","phase","error","message")},ensure_ascii=True)
        expect(panel.locator(".human-learning-report")).to_contain_text(report["run_id"][-8:])
        expect(panel.locator(".arena-summary")).to_be_visible()
        assert report["decision"] in ("promoted","rejected"),report
        assert report["candidate_sha256"] != parent
        assert report["arena"]["games"]==16
        assert report["data"]["selected_train_positions"]>=8
        current=context.request.get("/api/model").json()["checkpoint_sha256"]
        assert current==(report["candidate_sha256"] if report["decision"]=="promoted" else parent)
        with page.expect_download() as downloaded:
            panel.get_by_role("button",name="Tải báo cáo học JSON",exact=True).click()
        path=OUT/"human-learning-download.json"; downloaded.value.save_as(path)
        saved=json.loads(path.read_text(encoding="utf-8"))
        assert saved["run_id"]==report["run_id"] and len(saved["arena"]["matches"])==16
        assert "player_name" not in json.dumps(saved) and "token" not in json.dumps(saved)
        panel.screenshot(path=str(OUT/"human-learning-desktop.png"))
        print("PASS real history collection, training, 16-game gate, hot activation/retention, JSON report, and concurrent play",flush=True)
        panel.get_by_label("Tự học khi đủ ván mới").check()
        expect(panel.get_by_label("Tự học khi đủ ván mới")).to_be_checked()
        page.reload()
        page.get_by_role("button",name=re.compile("AI Lab")).first.click()
        expect(page.locator(".human-learning-panel").get_by_label("Tự học khi đủ ván mới")).to_be_checked()
        page.locator(".human-learning-panel").get_by_label("Tự học khi đủ ván mới").uncheck()
        mobile=browser.new_context(viewport={"width":390,"height":844},is_mobile=True,has_touch=True)
        phone=mobile.new_page(); phone.on("pageerror",lambda e:errors.append(str(e)))
        phone.goto("http://127.0.0.1:5173/")
        phone.get_by_role("button",name=re.compile("AI Lab")).first.click()
        expect(phone.locator(".human-learning-panel")).to_be_visible()
        assert phone.evaluate("document.documentElement.scrollWidth <= innerWidth"),"Human learning panel overflows on mobile"
        phone.locator(".human-learning-panel").screenshot(path=str(OUT/"human-learning-mobile.png"))
        assert not errors,json.dumps(errors,ensure_ascii=True)
        print("ALL HUMAN LEARNING BROWSER CHECKS PASSED",flush=True)
        browser.close()


if __name__=="__main__": main()
