import csv
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Sheet header/key tests do not need the optional Google client installed.
if "gspread" not in sys.modules:
    try:
        __import__("gspread")
    except ModuleNotFoundError:
        sys.modules["gspread"] = types.ModuleType("gspread")

import config as C  # noqa: E402
import push_to_sheet  # noqa: E402
import questionnaire as Q  # noqa: E402
import runner  # noqa: E402
import score  # noqa: E402


def valid_video_answer(q6_standard="human12"):
    q6 = {}
    for step in Q.ALGORITHM_STEPS:
        key = score.q6_key(step, q6_standard)
        q6[step] = Q.Q6_COLUMNS[0] if key is None else key
    return {
        "Q1": "Yes",
        "Q2": "Strongly agree",
        "Q3": "3",
        "Q4": "False",
        "Q5": 90,
        "Q6": q6,
        "Q7": "Yes",
        "Q8": "Nothing was unclear.",
        "Q9": "No change.",
    }


def valid_overall_answer():
    return {
        "Q17": 5, "Q18": 5, "Q19": 5, "Q20": 5,
        "Q21": "Nothing.", "Q22": "No change.", "Q23": "None.",
    }


class QuestionnaireTests(unittest.TestCase):
    def test_q6_protocol_thirteen_items_verbatim_and_ordered(self):
        # DO NOT MODIFY without written confirmation from the study PI.
        # This is the exact 13-item Q6 list as it appears in the original
        # Google Form response CSV.  Any reordering or renaming breaks
        # comparability with the human data.
        expected = [
            "Computing the center point",
            "Computing the edge points",
            "Dividing the space into slices",
            "Dividing the space into squares",
            "Reducing empty space",
            "Re-arranging empty space",
            "Finding gradients",
            "Finding outliers",
            "Finding empty space",
            "Introducing new empty spaces",
            "Computing new cluster centers",
            "Recursing into sub-regions",
            "Early termination",
        ]
        self.assertEqual(Q.ALGORITHM_STEPS, expected)

    def test_question_wording_matches_original_csv_headers(self):
        expected = {
            "Q1": "The video content was easy to understand.",
            "Q2": "The step-by-step animation helped me understand the clustering process.",
            "Q3": "How many clusters do you see at the end?",
            "Q4": "True or False: The user must specify the number of clusters before running the algorithm.",
            "Q5": "How much of the video content is understandable to you?",
            "Q6": "Which of the following steps did you observe in the animation and are part of the algorithm? (choose all that apply)",
            "Q7": "Do you think you could explain how this algorithm works to someone else?",
            "Q8": "What, if anything, was still unclear after watching the animation?",
            "Q9": "How would you improve the animation to make the process even clearer?",
        }
        self.assertEqual(
            {k: v["text"] for k, v in Q.PER_VIDEO_QUESTIONS.items()},
            expected,
        )
        self.assertNotIn("TODO", Q.per_video_prompt("test"))
        self.assertNotIn("TODO", Q.overall_prompt())

    def test_non_object_q6_warns_without_raising(self):
        answer = valid_video_answer()
        answer["Q6"] = "not a grid"
        warnings = Q.validate_per_video(answer)
        self.assertTrue(any("Q6 not an object" in w for w in warnings))
        self.assertEqual(answer["Q1"], "Yes")


class RunnerTests(unittest.TestCase):
    class FakeEndpoint:
        def __init__(self, errors):
            self.errors = list(errors)
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            error = self.errors.pop(0)
            if error is not None:
                raise error
            message = type("Message", (), {"content": "{}"})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    @staticmethod
    def fake_client(errors):
        endpoint = RunnerTests.FakeEndpoint(errors)
        chat = type("Chat", (), {"completions": endpoint})()
        return type("Client", (), {"chat": chat})(), endpoint

    def test_header_has_thirteen_item_q6_schema(self):
        header = runner.build_header()
        self.assertEqual(len(header), 107)
        self.assertIn("answer_constraint", header)
        self.assertIn("V1_Q6 [Reducing empty space]", header)
        self.assertIn("V1_Q6 [Finding empty space]", header)

    def test_retry_waits_only_between_transient_attempts(self):
        class ServerError(Exception):
            status_code = 500

        client, endpoint = self.fake_client(
            [ServerError("one"), ServerError("two"), ServerError("three")])
        with mock.patch.object(runner.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "after 3 attempt"):
                runner.chat(client, "model", [])
        self.assertEqual(endpoint.calls, 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [30, 60])

    def test_deterministic_400_is_not_retried(self):
        class ClientError(Exception):
            status_code = 400

        client, endpoint = self.fake_client([ClientError("bad request")])
        with mock.patch.object(runner.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "after 1 attempt"):
                runner.chat(client, "model", [])
        self.assertEqual(endpoint.calls, 1)
        sleep.assert_not_called()

    def test_none_content_fails_fast_with_finish_reason(self):
        # Thinking models (reasoning parser) return content=None when the
        # token budget is exhausted mid-reasoning.  That must surface as a
        # turn failure naming finish_reason, not flow into the JSON parser,
        # and must not burn transient-retry attempts (deterministic at T=0).
        message = type("Message", (), {"content": None})()
        choice = type("Choice", (), {"message": message,
                                     "finish_reason": "length"})()
        response = type("Response", (), {"choices": [choice]})()

        class FakeEndpoint:
            calls = 0

            def create(self, **kwargs):
                FakeEndpoint.calls += 1
                return response

        chat_ns = type("Chat", (), {"completions": FakeEndpoint()})()
        client = type("Client", (), {"chat": chat_ns})()
        with mock.patch.object(runner.time, "sleep") as sleep:
            with self.assertRaisesRegex(
                    RuntimeError, "finish_reason=length"):
                runner.chat(client, "model", [])
        self.assertEqual(FakeEndpoint.calls, 1)
        sleep.assert_not_called()

    def test_none_max_tokens_is_omitted_from_request(self):
        # MAX_TOKENS=None must mean "no request-side cap": the parameter
        # is left out entirely so vLLM allows generation up to the model's
        # context window. Sending max_tokens=None would be rejected.
        seen = []

        class FakeEndpoint:
            def create(self, **kwargs):
                seen.append(kwargs)
                message = type("Message", (), {"content": "{}"})()
                choice = type("Choice", (), {"message": message})()
                return type("Response", (), {"choices": [choice]})()

        chat_ns = type("Chat", (), {"completions": FakeEndpoint()})()
        client = type("Client", (), {"chat": chat_ns})()
        with mock.patch.object(C, "MAX_TOKENS", None):
            runner.chat(client, "model", [])
        self.assertNotIn("max_tokens", seen[0])
        with mock.patch.object(C, "MAX_TOKENS", 3000):
            runner.chat(client, "model", [])
        self.assertEqual(seen[1]["max_tokens"], 3000)

    def test_vendor_sampling_splits_native_and_extra_body(self):
        # top_k/min_p are not OpenAI SDK params — they must ride in
        # extra_body and merge with the structured-output payload; the
        # per-run seed goes through natively.
        seen = []

        class FakeEndpoint:
            def create(self, **kwargs):
                seen.append(kwargs)
                message = type("Message", (), {"content": "{}"})()
                choice = type("Choice", (), {"message": message})()
                return type("Response", (), {"choices": [choice]})()

        chat_ns = type("Chat", (), {"completions": FakeEndpoint()})()
        client = type("Client", (), {"chat": chat_ns})()
        sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                    "min_p": 0.0, "presence_penalty": 1.5}
        runner.chat(client, "m", [], schema={"type": "object"},
                    constraint_state={"mode": "structured_outputs"},
                    sampling=sampling, seed=2)
        kw = seen[0]
        self.assertEqual(kw["temperature"], 1.0)
        self.assertEqual(kw["top_p"], 0.95)
        self.assertEqual(kw["presence_penalty"], 1.5)
        self.assertEqual(kw["seed"], 2)
        self.assertNotIn("top_k", kw)
        self.assertEqual(kw["extra_body"]["top_k"], 20)
        self.assertEqual(kw["extra_body"]["min_p"], 0.0)
        self.assertIn("structured_outputs", kw["extra_body"])

    def test_every_model_declares_nonzero_vendor_sampling(self):
        # Greedy decoding (temperature=0) sends thinking models into
        # endless repetition; every registered model must carry its
        # vendor-recommended sampling.
        for tag, model in C.MODELS.items():
            with self.subTest(tag=tag):
                self.assertGreater(model["sampling"]["temperature"], 0)

    def test_bad_q6_does_not_erase_other_answers(self):
        video = valid_video_answer()
        video["Q6"] = "wrong type"
        replies = [json.dumps(video)] * 4 + [json.dumps(valid_overall_answer())]
        with tempfile.TemporaryDirectory() as td:
            raw = os.path.join(td, "raw")
            csv_path = os.path.join(td, "responses.csv")
            with mock.patch.object(C, "RAW_DIR", raw), \
                    mock.patch.object(C, "CSV_PATH", csv_path), \
                    mock.patch.object(runner, "chat", side_effect=replies):
                ok = runner.run_session(object(), "actual/model", "qwen3vl-8b", 1)
            self.assertTrue(ok)
            with open(csv_path, newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["V1_Q1"], "Yes")
            self.assertEqual(row["V1_Q6 [Computing the center point]"], "")
            self.assertIn("Q6 not an object", row["validation_warnings"])
            self.assertEqual(row["served_model"], "actual/model")

    def test_pilot_isolated_from_formal_csv(self):
        replies = ([json.dumps(valid_video_answer())] * 4
                   + [json.dumps(valid_overall_answer())])
        with tempfile.TemporaryDirectory() as td:
            formal = os.path.join(td, "formal", "responses.csv")
            pilot = os.path.join(td, "pilot", "responses.csv")
            with mock.patch.object(C, "CSV_PATH", formal), \
                    mock.patch.object(C, "PILOT_CSV_PATH", pilot), \
                    mock.patch.object(C, "PILOT_RAW_DIR", os.path.join(td, "pilot", "raw")), \
                    mock.patch.object(runner, "chat", side_effect=replies):
                ok = runner.run_session(
                    object(), "model", "qwen3vl-8b", 1, pilot=True)
            self.assertTrue(ok)
            self.assertTrue(os.path.exists(pilot))
            self.assertFalse(os.path.exists(formal))

    def test_dry_run_never_writes_a_csv(self):
        with tempfile.TemporaryDirectory() as td:
            formal = os.path.join(td, "responses.csv")
            with mock.patch.object(C, "CSV_PATH", formal), \
                    mock.patch.object(C, "RAW_DIR", os.path.join(td, "raw")):
                ok = runner.run_session(
                    object(), "model", "qwen3vl-8b", 1, dry=True)
            self.assertTrue(ok)
            self.assertFalse(os.path.exists(formal))

    def test_each_turn_retains_all_prior_videos_in_one_conversation(self):
        video_reply = json.dumps(valid_video_answer())
        overall_reply = json.dumps(valid_overall_answer())
        video_counts = []
        final_roles = []

        def inspect_chat(client, served_model, messages, **kwargs):
            count = 0
            for message in messages:
                content = message.get("content")
                if isinstance(content, list):
                    count += sum(part.get("type") == "video_url"
                                 for part in content)
            video_counts.append(count)
            if len(video_counts) == 5:
                final_roles.extend(message["role"] for message in messages)
                self.assertIsInstance(messages[-1]["content"], str)
                self.assertIn("all four videos", messages[-1]["content"])
            return overall_reply if len(video_counts) == 5 else video_reply

        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(C, "CSV_PATH", os.path.join(td, "responses.csv")), \
                    mock.patch.object(C, "RAW_DIR", os.path.join(td, "raw")), \
                    mock.patch.object(runner, "chat", side_effect=inspect_chat):
                ok = runner.run_session(
                    object(), "model", "qwen3vl-8b", 1)
        self.assertTrue(ok)
        self.assertEqual(video_counts, [1, 2, 3, 4, 4])
        self.assertEqual(final_roles, [
            "system", "user", "assistant", "user", "assistant",
            "user", "assistant", "user", "assistant", "user",
        ])

    def test_existing_mismatched_csv_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "responses.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["old", "schema"])
            with self.assertRaisesRegex(RuntimeError, "schema differs"):
                runner.append_csv({}, path)


class ScoreTests(unittest.TestCase):
    def write_rows(self, rows):
        temp = tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", delete=False)
        with temp:
            writer = csv.DictWriter(temp, fieldnames=runner.build_header())
            writer.writeheader()
            writer.writerows(rows)
        self.addCleanup(lambda: os.path.exists(temp.name) and os.unlink(temp.name))
        return temp.name

    def flattened(self, tag, answer):
        per_video = {v["id"]: dict(answer, Q3=v["q3_expected"])
                     for v in C.VIDEOS}
        meta = {k: "" for k in runner.METADATA_COLS}
        meta.update({"model_tag": tag, "run_id": 1, "timestamp": "t"})
        return runner.flatten_row(
            meta, per_video, valid_overall_answer(), [])

    def test_full_score_arithmetic_for_both_q6_keys(self):
        human_row = self.flattened("human-perfect", valid_video_answer("human12"))
        design_row = self.flattened("design-perfect", valid_video_answer("design13"))
        result = score.score_rows(self.write_rows([human_row, design_row]))
        self.assertEqual(
            (result["human-perfect"]["q6_h12_correct"],
             result["human-perfect"]["q6_h12_total"]), (48, 48))
        self.assertEqual(
            (result["human-perfect"]["q6_d13_correct"],
             result["human-perfect"]["q6_d13_total"]), (48, 52))
        self.assertEqual(
            (result["design-perfect"]["q6_d13_correct"],
             result["design-perfect"]["q6_d13_total"]), (52, 52))
        self.assertEqual(
            (result["design-perfect"]["q6_h12_correct"],
             result["design-perfect"]["q6_h12_total"]), (44, 48))

    def test_missing_answers_are_excluded_and_format_rate_records_failure(self):
        row = {k: "" for k in runner.build_header()}
        row.update({"model_tag": "missing", "run_id": "1", "timestamp": "t"})
        result = score.score_rows(self.write_rows([row]))["missing"]
        self.assertEqual(result["q3_total"], 0)
        self.assertEqual(result["q4_total"], 0)
        self.assertEqual(result["q6_d13_total"], 0)
        self.assertEqual(result["video_format_correct"], 0)
        self.assertEqual(result["video_format_total"], 4)
        self.assertEqual(result["overall_format_correct"], 0)


class SheetTests(unittest.TestCase):
    def test_extra_manual_columns_are_allowed_only_at_end(self):
        csv_header = ["a", "b"]
        self.assertTrue(push_to_sheet.header_is_compatible(
            ["a", "b", "manual note"], csv_header))
        self.assertFalse(push_to_sheet.header_is_compatible(
            ["a", "manual note", "b"], csv_header))


class StructuredOutputTests(unittest.TestCase):
    def test_per_video_schema_matches_form_options(self):
        s = Q.per_video_schema()
        self.assertEqual(s["properties"]["Q3"]["enum"],
                         Q.CLUSTER_COUNTS)
        self.assertIn("I couldn't tell", s["properties"]["Q3"]["enum"])
        q6 = s["properties"]["Q6"]
        self.assertEqual(sorted(q6["properties"]),
                         sorted(Q.ALGORITHM_STEPS))
        self.assertEqual(q6["required"], list(Q.ALGORITHM_STEPS))
        for p in q6["properties"].values():
            self.assertEqual(p["enum"], Q.Q6_COLUMNS)
        self.assertEqual(s["properties"]["Q5"]["minimum"], 0)
        self.assertEqual(s["properties"]["Q5"]["maximum"], 100)

    def test_overall_schema_bounds(self):
        s = Q.overall_schema()
        for k in ["Q17", "Q18", "Q19", "Q20"]:
            self.assertEqual(s["properties"][k]["minimum"], 1)
            self.assertEqual(s["properties"][k]["maximum"], 5)

    def test_mode_negotiation_falls_back_and_sticks(self):
        calls = []

        class FakeErr(Exception):
            status_code = 400

        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kw):
                        calls.append(kw)
                        if "response_format" in kw:
                            raise FakeErr("json_schema unsupported")
                        msg = types.SimpleNamespace(content="{}")
                        choice = types.SimpleNamespace(message=msg)
                        return types.SimpleNamespace(choices=[choice])

        state = {}
        with mock.patch.object(runner.time, "sleep") as slept:
            out = runner.chat(FakeClient(), "m", [],
                              schema={"type": "object"},
                              constraint_state=state)
        self.assertEqual(out, "{}")
        self.assertEqual(state["mode"], "structured_outputs")
        self.assertFalse(slept.called)  # fallback consumed no retry sleeps
        self.assertIn("extra_body", calls[-1])
        self.assertIn("structured_outputs", calls[-1]["extra_body"])
        # second turn reuses the negotiated mode without re-probing
        calls.clear()
        runner.chat(FakeClient(), "m", [], schema={"type": "object"},
                    constraint_state=state)
        self.assertEqual(len(calls), 1)
        self.assertIn("extra_body", calls[0])

    def test_open_questions_require_non_empty_answers(self):
        p = Q.per_video_prompt("Video 1: Blobs")
        self.assertIn("must provide a non-empty answer", p)
        o = Q.overall_prompt()
        self.assertIn("required non-empty free text", o.lower())

    def test_blank_open_answers_generate_validation_warnings(self):
        video = valid_video_answer()
        video["Q8"] = "   "
        self.assertTrue(any("Q8 not non-empty text" in warning
                            for warning in Q.validate_per_video(video)))
        overall = valid_overall_answer()
        overall["Q21"] = ""
        self.assertTrue(any("Q21 not non-empty text" in warning
                            for warning in Q.validate_overall(overall)))

    def test_422_also_negotiates_the_next_structured_mode(self):
        class Unprocessable(Exception):
            status_code = 422

        client, endpoint = RunnerTests.fake_client(
            [Unprocessable("unsupported response_format"), None])
        state = {}
        out = runner.chat(client, "m", [], schema={"type": "object"},
                          constraint_state=state)
        self.assertEqual(out, "{}")
        self.assertEqual(endpoint.calls, 2)
        self.assertEqual(state["mode"], "structured_outputs")


class ServerScriptTests(unittest.TestCase):
    def test_node03_gpu_mapping_is_written_into_formal_scripts(self):
        expected = {
            "qwen3vl_8b.sh": "CUDA_VISIBLE_DEVICES=3",
            "qwen35_9b.sh": "CUDA_VISIBLE_DEVICES=3",
            "minicpmv45.sh": "CUDA_VISIBLE_DEVICES=3",
            "internvl35_38b.sh": "CUDA_VISIBLE_DEVICES=4,5",
            "glm46v.sh": "CUDA_VISIBLE_DEVICES=4,5,6,7",
        }
        for filename, mapping in expected.items():
            with self.subTest(filename=filename):
                text = (ROOT / "serve" / filename).read_text(encoding="utf-8")
                self.assertIn(mapping, text)

    def test_legacy_scripts_are_not_registered_models(self):
        registered = {model["serve_script"] for model in C.MODELS.values()}
        self.assertNotIn("serve/glm45v.sh", registered)
        self.assertNotIn("serve/qwen3vl_235b_fp8.sh", registered)

    def test_q6_scoring_human12_design13_semantics(self):
        # DO NOT MODIFY without written confirmation from the study PI.
        # human12 is the primary human-comparable score: 12 items
        # (Finding empty space excluded, Early termination scored No).
        # design13 is the sensitivity analysis: all 13 items
        # (Early termination and Finding empty space both scored Yes).
        self.assertIsNone(score.q6_key("Finding empty space", "human12"),
                          "human12 must exclude Finding empty space")
        self.assertEqual(score.q6_key("Early termination", "human12"), Q.Q6_COLUMNS[1],
                         "human12 must score Early termination as No")
        self.assertEqual(score.q6_key("Finding empty space", "design13"), Q.Q6_COLUMNS[0],
                         "design13 must score Finding empty space as Yes")
        self.assertEqual(score.q6_key("Early termination", "design13"), Q.Q6_COLUMNS[0],
                         "design13 must score Early termination as Yes")
        for step in Q.DISTRACTORS:
            self.assertEqual(score.q6_key(step, "human12"), Q.Q6_COLUMNS[1])
            self.assertEqual(score.q6_key(step, "design13"), Q.Q6_COLUMNS[1])


class SingleVideoPromptTests(unittest.TestCase):
    def test_strip_video_parts_replaces_videos_and_keeps_text(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user",
             "content": [runner.video_content_part("videos/v1.mp4"),
                         {"type": "text", "text": "Q1-Q9"}]},
            {"role": "assistant", "content": "{}"},
        ]
        stripped = runner.strip_video_parts(messages)
        # original untouched (no mutation)
        self.assertEqual(messages[1]["content"][0]["type"], "video_url")
        self.assertEqual(stripped[0], messages[0])
        self.assertEqual(stripped[2], messages[2])
        parts = stripped[1]["content"]
        self.assertTrue(all(p["type"] == "text" for p in parts))
        self.assertEqual(parts[0]["text"], runner.VIDEO_REMOVED_NOTE)
        self.assertEqual(parts[1]["text"], "Q1-Q9")

    def test_glm46v_is_registered_as_single_video_model(self):
        # vLLM's GLM-4V implementation caps video at 1 per prompt; the
        # runner must strip earlier videos for this model or every
        # session dies at V2 with 'At most 1 video(s) may be provided'.
        self.assertEqual(C.MODELS["glm-4.6v"].get("max_videos_per_prompt"), 1)


if __name__ == "__main__":
    unittest.main()
