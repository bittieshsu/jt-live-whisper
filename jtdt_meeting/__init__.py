"""會議摘要核心：從 jt-doc-tools（JTDT）原樣搬來（2026-09-28，JTDT v1.16.25 的 app/core/）。

**這個資料夾裡的模組不要在 jtlw 這邊改**：它們與 JTDT 是同一份（`tools/test_jtdt_modules_in_sync.py` 比對），
要改就在 JTDT 改完再同步過來，兩個專案的會議摘要才會一直是同一個品質。
jtlw 這邊的接法（呼叫 LLM、輸出檔案、API）在 `translate_meeting.py` 與 `jtlw_api/`。
"""
