#!/usr/bin/env python3
"""Interactively review BMW Template and EfficientAD NG evidence by part."""

from __future__ import annotations

import argparse
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Sequence

from PIL import Image, ImageTk


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from bmw_inspection.lab.ng_review import ReviewCase, ReviewDataset  # noqa: E402


def _shared_workspace_root(repo_root: Path) -> Path:
    if repo_root.parent.name == ".worktrees":
        return repo_root.parent.parent
    return repo_root


DEFAULT_REVIEW_CSV = (
    _shared_workspace_root(REPO_ROOT)
    / "results/bmw_template_efficientad_ng_review_left_20260816_v1/review_cases.csv"
)

DECISION_COLORS = {
    "": "#64748b",
    "误判": "#16a34a",
    "真实缺陷": "#dc2626",
    "不确定": "#d97706",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the local NG reviewer command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查复核包并输出数量，不打开窗口。",
    )
    return parser


class NgReviewApp:
    """Tkinter application for reviewing detector problems one part at a time."""

    def __init__(self, root: tk.Tk, dataset: ReviewDataset) -> None:
        self.root = root
        self.dataset = dataset
        self.group_index = 0
        self.case_index = 0
        self._image_refs: list[ImageTk.PhotoImage] = []
        self._case_frames: list[tk.Frame] = []
        self._decision_vars: dict[str, tk.StringVar] = {}
        self._note_vars: dict[str, tk.StringVar] = {}
        self._zoom_window: tk.Toplevel | None = None

        root.title("BMW Template / EfficientAD 缺陷复核")
        root.geometry("1560x940")
        root.minsize(1180, 720)
        root.configure(bg="#eef2f7")
        root.option_add("*Font", ("Noto Sans CJK SC", 11))

        self._build_layout()
        self._bind_shortcuts()
        self._render_group()

    def _build_layout(self) -> None:
        header = tk.Frame(self.root, bg="#0f172a", padx=18, pady=13)
        header.pack(fill="x")
        tk.Label(
            header,
            text="BMW 误判复核",
            bg="#0f172a",
            fg="white",
            font=("Noto Sans CJK SC", 18, "bold"),
        ).pack(side="left")
        self.progress_label = tk.Label(header, bg="#0f172a", fg="#cbd5e1")
        self.progress_label.pack(side="left", padx=24)
        self.save_label = tk.Label(header, text="", bg="#0f172a", fg="#86efac")
        self.save_label.pack(side="right")

        body = tk.Frame(self.root, bg="#eef2f7")
        body.pack(fill="both", expand=True)

        sidebar = tk.Frame(body, width=325, bg="#f8fafc", padx=10, pady=10)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        tk.Label(
            sidebar,
            text="零件列表",
            bg="#f8fafc",
            fg="#0f172a",
            font=("Noto Sans CJK SC", 13, "bold"),
        ).pack(anchor="w", pady=(0, 7))
        list_frame = tk.Frame(sidebar, bg="#f8fafc")
        list_frame.pack(fill="both", expand=True)
        self.part_list = tk.Listbox(
            list_frame,
            activestyle="none",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground="#cbd5e1",
            selectbackground="#2563eb",
            selectforeground="white",
            bg="white",
            fg="#1e293b",
        )
        list_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.part_list.yview)
        self.part_list.configure(yscrollcommand=list_scroll.set)
        self.part_list.pack(side="left", fill="both", expand=True)
        list_scroll.pack(side="right", fill="y")
        self.part_list.bind("<<ListboxSelect>>", self._on_part_selected)

        content = tk.Frame(body, bg="#eef2f7")
        content.pack(side="left", fill="both", expand=True)

        toolbar = tk.Frame(content, bg="#e2e8f0", padx=14, pady=9)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="← 上一件", command=lambda: self._move_group(-1)).pack(side="left")
        ttk.Button(toolbar, text="下一件 →", command=lambda: self._move_group(1)).pack(side="left", padx=(7, 18))
        tk.Label(toolbar, text="将本零件未选择项全部标为：", bg="#e2e8f0", fg="#334155").pack(side="left")
        for decision in ("误判", "真实缺陷", "不确定"):
            tk.Button(
                toolbar,
                text=decision,
                command=lambda value=decision: self._batch_decide(value),
                bg=DECISION_COLORS[decision],
                fg="white",
                activebackground=DECISION_COLORS[decision],
                activeforeground="white",
                relief="flat",
                padx=12,
                pady=5,
            ).pack(side="left", padx=4)
        ttk.Button(toolbar, text="保存", command=self._save).pack(side="right")

        self.canvas = tk.Canvas(content, bg="#eef2f7", highlightthickness=0)
        content_scroll = ttk.Scrollbar(content, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=content_scroll.set)
        content_scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.cards = tk.Frame(self.canvas, bg="#eef2f7", padx=14, pady=10)
        self.cards_window = self.canvas.create_window((0, 0), anchor="nw", window=self.cards)
        self.cards.bind(
            "<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.bind(
            "<Configure>", lambda event: self.canvas.itemconfigure(self.cards_window, width=event.width)
        )
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _bind_shortcuts(self) -> None:
        self.root.bind_all("<Control-s>", lambda _event: self._save())
        self.root.bind_all("<Left>", lambda event: self._shortcut_group(event, -1))
        self.root.bind_all("<Right>", lambda event: self._shortcut_group(event, 1))
        self.root.bind_all("<Up>", lambda event: self._shortcut_case(event, -1))
        self.root.bind_all("<Down>", lambda event: self._shortcut_case(event, 1))
        self.root.bind_all("1", lambda event: self._shortcut_decision(event, "误判"))
        self.root.bind_all("2", lambda event: self._shortcut_decision(event, "真实缺陷"))
        self.root.bind_all("3", lambda event: self._shortcut_decision(event, "不确定"))
        self.root.bind_all("<Escape>", lambda _event: self._close_zoom())

    def _render_group(self) -> None:
        for child in self.cards.winfo_children():
            child.destroy()
        self._image_refs.clear()
        self._case_frames.clear()
        self._decision_vars.clear()
        self._note_vars.clear()
        self._refresh_sidebar()
        group = self.dataset.groups[self.group_index]

        title = tk.Label(
            self.cards,
            text=f"{group.capture_id}　·　共 {len(group.cases)} 个问题",
            bg="#eef2f7",
            fg="#0f172a",
            anchor="w",
            font=("Noto Sans CJK SC", 16, "bold"),
        )
        title.pack(fill="x", pady=(0, 10))
        for index, case in enumerate(group.cases):
            self._render_case(case, index)
        self.case_index = min(self.case_index, max(0, len(group.cases) - 1))
        self._highlight_case()
        self.canvas.yview_moveto(0)
        self._refresh_progress()

    def _render_case(self, case: ReviewCase, index: int) -> None:
        card = tk.Frame(
            self.cards,
            bg="white",
            highlightthickness=2,
            highlightbackground="#cbd5e1",
            padx=12,
            pady=10,
        )
        card.pack(fill="x", pady=(0, 12))
        self._case_frames.append(card)
        card.bind("<Button-1>", lambda _event, value=index: self._select_case(value))

        branch = case.row.get("branch_cn") or ("EfficientAD" if case.row["branch"] == "efficientad" else "模板匹配")
        view = case.row.get("view_cn") or case.row["view_id"]
        score = case.row.get("score", "-")
        threshold = case.row.get("threshold", "-")
        header_text = f"{case.case_id}　{branch}　{view}　异常分数 {score} / 阈值 {threshold}"
        header = tk.Label(
            card,
            text=header_text,
            bg="white",
            fg="#0f172a",
            anchor="w",
            font=("Noto Sans CJK SC", 13, "bold"),
        )
        header.pack(fill="x")
        header.bind("<Button-1>", lambda _event, value=index: self._select_case(value))

        reason = case.row.get("reason", "")
        if reason:
            tk.Label(
                card,
                text=f"算法原因：{reason}",
                bg="white",
                fg="#475569",
                anchor="w",
                justify="left",
                wraplength=1100,
            ).pack(fill="x", pady=(3, 7))

        image_label = self._make_panel_label(card, Path(case.row["panel_path"]))
        image_label.pack(fill="x", pady=(0, 9))
        image_label.bind("<Button-1>", lambda _event, value=index: self._select_case(value))
        image_label.bind("<Double-Button-1>", lambda _event, path=case.row["panel_path"]: self._zoom(path))

        controls = tk.Frame(card, bg="white")
        controls.pack(fill="x")
        tk.Label(controls, text="人工结论：", bg="white", fg="#334155").pack(side="left")
        decision_var = tk.StringVar(value=case.decision)
        self._decision_vars[case.case_id] = decision_var
        for decision in ("误判", "真实缺陷", "不确定"):
            ttk.Radiobutton(
                controls,
                text=decision,
                value=decision,
                variable=decision_var,
                command=lambda item=case, value=decision: self._decide(item, value),
            ).pack(side="left", padx=(2, 12))
        tk.Label(controls, text="备注：", bg="white", fg="#334155").pack(side="left", padx=(12, 4))
        note_var = tk.StringVar(value=case.review_note)
        self._note_vars[case.case_id] = note_var
        note_entry = ttk.Entry(controls, textvariable=note_var)
        note_entry.pack(side="left", fill="x", expand=True)
        note_entry.bind("<FocusIn>", lambda _event, value=index: self._select_case(value))
        note_entry.bind("<FocusOut>", lambda _event, item=case: self._save_note(item))
        note_entry.bind("<Return>", lambda _event, item=case: self._save_note(item))

    def _make_panel_label(self, parent: tk.Widget, path: Path) -> tk.Label:
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
            image.thumbnail((1120, 330), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self._image_refs.append(photo)
            return tk.Label(parent, image=photo, bg="#0f172a", cursor="hand2")
        except (OSError, ValueError) as error:
            return tk.Label(
                parent,
                text=f"图片无法读取：{path}\n{error}",
                bg="#fee2e2",
                fg="#991b1b",
                pady=30,
            )

    def _refresh_sidebar(self) -> None:
        self.part_list.delete(0, "end")
        for group in self.dataset.groups:
            reviewed = sum(bool(case.decision) for case in group.cases)
            marker = "✓" if reviewed == len(group.cases) else "○"
            self.part_list.insert("end", f"{marker} {group.capture_id}  ({reviewed}/{len(group.cases)})")
        self.part_list.selection_set(self.group_index)
        self.part_list.activate(self.group_index)
        self.part_list.see(self.group_index)

    def _refresh_progress(self) -> None:
        progress = self.dataset.progress
        self.progress_label.configure(
            text=(
                f"零件 {progress.completed_capture_count}/{progress.capture_count}　"
                f"问题 {progress.reviewed_case_count}/{progress.case_count}"
            )
        )

    def _on_part_selected(self, _event: tk.Event[tk.Misc]) -> None:
        selection = self.part_list.curselection()
        if selection and selection[0] != self.group_index:
            self.group_index = selection[0]
            self.case_index = 0
            self._render_group()

    def _select_case(self, index: int) -> None:
        self.case_index = index
        self._highlight_case()

    def _highlight_case(self) -> None:
        for index, frame in enumerate(self._case_frames):
            frame.configure(highlightbackground="#2563eb" if index == self.case_index else "#cbd5e1")

    def _decide(self, case: ReviewCase, decision: str) -> None:
        self.dataset.set_decision(case.case_id, decision)
        self._save()

    def _save_note(self, case: ReviewCase) -> str:
        self.dataset.set_note(case.case_id, self._note_vars[case.case_id].get())
        self._save()
        return "break"

    def _batch_decide(self, decision: str) -> None:
        capture_id = self.dataset.groups[self.group_index].capture_id
        changed = self.dataset.set_remaining_for_capture(capture_id, decision)
        for case in self.dataset.groups[self.group_index].cases:
            self._decision_vars[case.case_id].set(case.decision)
        self._save()
        self.save_label.configure(text=f"已标记 {changed} 个未选择问题")

    def _save(self) -> None:
        try:
            self.dataset.save()
        except OSError as error:
            self.save_label.configure(text="保存失败", fg="#fca5a5")
            messagebox.showerror("保存失败", f"复核结果没有写入：\n{error}", parent=self.root)
            return
        self.save_label.configure(text="已自动保存", fg="#86efac")
        self._refresh_sidebar()
        self._refresh_progress()

    def _move_group(self, offset: int) -> str:
        target = max(0, min(len(self.dataset.groups) - 1, self.group_index + offset))
        if target != self.group_index:
            self.group_index = target
            self.case_index = 0
            self._render_group()
        return "break"

    def _move_case(self, offset: int) -> str:
        cases = self.dataset.groups[self.group_index].cases
        self.case_index = max(0, min(len(cases) - 1, self.case_index + offset))
        self._highlight_case()
        frame = self._case_frames[self.case_index]
        self.canvas.yview_moveto(max(0.0, frame.winfo_y() / max(1, self.cards.winfo_height())))
        return "break"

    def _shortcut_decision(self, event: tk.Event[tk.Misc], decision: str) -> str | None:
        if isinstance(event.widget, (tk.Entry, ttk.Entry)):
            return None
        case = self.dataset.groups[self.group_index].cases[self.case_index]
        self._decision_vars[case.case_id].set(decision)
        self._decide(case, decision)
        return "break"

    def _shortcut_group(self, event: tk.Event[tk.Misc], offset: int) -> str | None:
        if isinstance(event.widget, (tk.Entry, ttk.Entry)):
            return None
        return self._move_group(offset)

    def _shortcut_case(self, event: tk.Event[tk.Misc], offset: int) -> str | None:
        if isinstance(event.widget, (tk.Entry, ttk.Entry)):
            return None
        return self._move_case(offset)

    def _on_mousewheel(self, event: tk.Event[tk.Misc]) -> str:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")
        return "break"

    def _zoom(self, path: str) -> None:
        self._close_zoom()
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
        except (OSError, ValueError) as error:
            messagebox.showerror("图片无法读取", f"{path}\n{error}", parent=self.root)
            return
        window = tk.Toplevel(self.root)
        self._zoom_window = window
        window.title(Path(path).name)
        window.configure(bg="#020617")
        max_width = max(800, window.winfo_screenwidth() - 80)
        max_height = max(500, window.winfo_screenheight() - 120)
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(window, image=photo, bg="#020617")
        label.image = photo  # type: ignore[attr-defined]
        label.pack(fill="both", expand=True)
        window.geometry(f"{image.width}x{image.height}")
        window.bind("<Escape>", lambda _event: self._close_zoom())
        window.protocol("WM_DELETE_WINDOW", self._close_zoom)
        window.focus_force()

    def _close_zoom(self) -> None:
        if self._zoom_window is not None and self._zoom_window.winfo_exists():
            self._zoom_window.destroy()
        self._zoom_window = None


def _check_summary(dataset: ReviewDataset) -> dict[str, int | str]:
    progress = dataset.progress
    return {
        "status": "ok",
        "review_csv": str(dataset.path),
        "capture_count": progress.capture_count,
        "case_count": progress.case_count,
        "template_count": sum(case.row["branch"] == "template" for case in dataset.cases),
        "efficientad_count": sum(case.row["branch"] == "efficientad" for case in dataset.cases),
        "reviewed_case_count": progress.reviewed_case_count,
        "missing_panel_count": sum(not Path(case.row["panel_path"]).is_file() for case in dataset.cases),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Load the review package and run either validation or the GUI."""
    args = build_parser().parse_args(argv)
    try:
        dataset = ReviewDataset.load(args.review_csv)
    except (OSError, ValueError) as error:
        print(f"BMW缺陷复核器启动失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    if args.check:
        print(json.dumps(_check_summary(dataset), ensure_ascii=False, sort_keys=True))
        return 0
    root = tk.Tk()
    NgReviewApp(root, dataset)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
