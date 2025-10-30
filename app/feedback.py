"""Feedback logging and analytics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .config import AppConfig


@dataclass
class FeedbackEntry:
    timestamp: str
    query: str
    answer: str
    confidence_pct: int
    confidence_level: str
    num_sources: int
    top_sources: List[str]
    retriever_type: str
    feedback: str
    correction: str
    attempts: int
    was_refined: bool

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


class FeedbackSystem:
    """Collects and analyses user feedback on RAG answers."""

    def __init__(self, log_path: Path | None = None) -> None:
        paths = AppConfig.get().paths
        self.log_path = log_path or paths.feedback_log
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_feedback(self, result: Dict, feedback_type: str, correction: str = "") -> None:
        entry = FeedbackEntry(
            timestamp=datetime.now().isoformat(),
            query=result.get("query", ""),
            answer=result.get("answer", ""),
            confidence_pct=result.get("confidence_pct", 0),
            confidence_level=result.get("confidence_level", ""),
            num_sources=result.get("num_sources", 0),
            top_sources=[src["source"] for src in result.get("sources", [])[:3]],
            retriever_type=result.get("retriever_type", "unknown"),
            feedback=feedback_type,
            correction=correction,
            attempts=result.get("attempts", 1),
            was_refined=result.get("final_query") != result.get("query"),
        )
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(entry.to_json() + "\n")

    def get_all_feedback(self) -> List[Dict]:
        if not self.log_path.exists():
            return []
        feedback: List[Dict] = []
        with self.log_path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    feedback.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return feedback

    def analyze_feedback(self) -> Dict:
        feedback = self.get_all_feedback()
        if not feedback:
            return {"error": "No feedback data available"}

        total = len(feedback)
        helpful = sum(1 for item in feedback if item["feedback"] == "helpful")
        incorrect = sum(1 for item in feedback if item["feedback"] == "incorrect")

        high_conf_correct = [item for item in feedback if item["feedback"] == "helpful" and item["confidence_pct"] >= 75]
        high_conf_incorrect = [item for item in feedback if item["feedback"] == "incorrect" and item["confidence_pct"] >= 75]
        low_conf_correct = [item for item in feedback if item["feedback"] == "helpful" and item["confidence_pct"] < 60]
        low_conf_incorrect = [item for item in feedback if item["feedback"] == "incorrect" and item["confidence_pct"] < 60]

        refined_queries = [item for item in feedback if item.get("was_refined")]
        refined_helpful = sum(1 for item in refined_queries if item["feedback"] == "helpful")

        analysis = {
            "total_feedback": total,
            "satisfaction_rate": helpful / total * 100 if total else 0,
            "incorrect_rate": incorrect / total * 100 if total else 0,
            "confidence_calibration": {
                "high_conf_accurate": len(high_conf_correct),
                "high_conf_wrong": len(high_conf_incorrect),
                "overconfidence_rate": len(high_conf_incorrect)
                / max(len(high_conf_correct) + len(high_conf_incorrect), 1)
                * 100,
                "low_conf_accurate": len(low_conf_correct),
                "low_conf_wrong": len(low_conf_incorrect),
                "underconfidence_rate": len(low_conf_correct)
                / max(len(low_conf_correct) + len(low_conf_incorrect), 1)
                * 100,
            },
            "query_refinement": {
                "total_refined": len(refined_queries),
                "refined_helpful": refined_helpful,
                "refinement_success_rate": refined_helpful / len(refined_queries) * 100 if refined_queries else 0,
            },
            "recommendations": [],
        }

        if analysis["confidence_calibration"]["overconfidence_rate"] > 20:
            analysis["recommendations"].append(
                "⚠️  System is overconfident - consider raising HIGH confidence threshold from 75% to 80%"
            )
        if analysis["confidence_calibration"]["underconfidence_rate"] > 30:
            analysis["recommendations"].append(
                "📈 System is underconfident - consider lowering HIGH confidence threshold from 75% to 70%"
            )
        if analysis["query_refinement"]["refinement_success_rate"] < 50 and len(refined_queries) > 5:
            analysis["recommendations"].append(
                "🔄 Query refinement not helping - consider disabling auto_refine by default"
            )
        if analysis["incorrect_rate"] > 15:
            analysis["recommendations"].append(
                "❌ High incorrect rate - review top error cases and improve retrieval/prompting"
            )
        return analysis

    def get_problem_queries(self, feedback_type: str = "incorrect", limit: int = 10) -> List[Dict]:
        feedback = self.get_all_feedback()
        problems = [item for item in feedback if item["feedback"] == feedback_type]
        problems.sort(key=lambda x: x["confidence_pct"], reverse=True)
        return problems[:limit]


__all__ = ["FeedbackSystem"]
