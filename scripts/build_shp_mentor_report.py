"""Build the SHP mentor report from reviewed, saved experiment evidence (#20/#21)."""

from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

OUT = Path("docs/reports/SHP_predictive_modeling_report_for_mentor.docx")
PAGES = [
    [
        ("title", "SHP predictive modeling assessment"),
        ("p", "Why reliable advance fault warning has not yet been demonstrated"),
        ("p", "Prepared for mentor review | 1 November 2026"),
        (
            "p",
            "Project title: A Machine Learning–Based Early Fault Warning System for Renewable Energy Equipment",
        ),
        ("h", "Main conclusion"),
        (
            "p",
            "The tested SHP configurations have not demonstrated reliable early fault warning under the project’s chronological evaluation and alarm constraints. Several conservative models miss most or all eligible faults. The strongest sequence candidate detects more events, but produces frequent, short alarms and lacks independent confirmation. This is evidence that the tested approaches are not yet successful for the intended task; it is not proof that SHP contains no predictive information.",
        ),
        ("h", "What success means in this project"),
        (
            "p",
            "The objective is to warn before a recorded fault, within a six-hour look-ahead window, using only information available at the prediction time. It is not remaining useful life estimation. The six-hour horizon and 2% selected-nominal point-alert budget are research choices that still require engineering and supervisor agreement. A warning 44 minutes before a fault qualifies within this window; it does not mean the model predicts faults six hours in advance.",
        ),
        (
            "p",
            "A useful candidate must add adequately covered fault groups without losing baseline detections and remain within the declared alarm budget. For the main comparisons, an actionable alarm requires three consecutive threshold exceedances at valid sampling intervals. Single-reading results are reported separately.",
        ),
        ("h", "Evidence covered"),
        (
            "p",
            "This report consolidates the saved LSTM baseline, trend experiment, corrected supervised classifier, earlier chronological backtests, Isolation Forest and Extended Isolation Forest comparisons, conditional-vibration model, and sequence-pattern experiments. It summarizes existing results; no model was retrained for this report. Local reports and aggregate evidence are identified on page 5.",
        ),
        (
            "p",
            "GitHub issues #19, #20 and #21 were read on 1 November 2026 and remain open at review time. Their bodies predate some completed local studies. Local experiment completion is therefore distinguished from publication, acceptance review and independent validation.",
        ),
    ],
    [
        ("h", "Dataset and evaluation limitations"),
        (
            "p",
            "SHP is small hydroelectric plant telemetry. The source contains six monitored channels and 59 fault timestamps. The local timestamp union has 52,754 unique observations: 47,857 publisher-selected normal readings and 4,897 pre-fault readings. The publication formed its test file from the twelve hours before faults; these files are not an untouched chronological train–test split. [1, 2]",
        ),
        ("h", "Fault timestamps are weak prediction labels"),
        (
            "p",
            "The local fault log lacks failure modes, causes, repairs, event duration and a verified distinction between physical onset and trip or record-entry time. A label meaning “a recorded fault occurs within six hours” is a proximity label, not proof that every preceding reading shows degradation. Likewise, a negative label does not prove equipment health. Pooling potentially different mechanisms may make the target inconsistent, but that explanation cannot be confirmed without event records. [2]",
        ),
        ("h", "Many readings do not mean many independent failures"),
        (
            "p",
            "Adjacent readings share history and labels. Overlapping six-hour fault windows are grouped to reduce double counting, although groups are not confirmed independent physical failures. After corrected supervised preparation, fitting contains 25,855 endpoints and 1,473 positive labels, but usable fault coverage spans only 15 fitting groups. January and February–April each provide only three eligible groups for this experiment. [3]",
        ),
        (
            "table",
            (
                ["Period", "Fault timestamps", "Raw coverage", "Sequence coverage"],
                [
                    ["Before February 2019", "39", "28", "21"],
                    ["February–April", "9", "6", "3"],
                    ["May onward", "11", "5", "4"],
                ],
            ),
        ),
        (
            "p",
            "Table 1. Full-union availability inventory, not model detection results. Coverage requires at least 80% of expected five-minute slots in the preceding six hours and a reading within ten minutes of the event. Sequence availability additionally needs seven readings with 150–450-second steps. Partition boundaries and feature warm-up can reduce these counts. [2]",
        ),
        ("h", "Gaps and missing context limit what can be learned"),
        (
            "p",
            "Longer trend histories lose usable events when gaps force warm-up resets. Interpolation cannot create real precursor evidence. Author-code review maps V1–V4 to vibration, V5 to lubrication inflow and V6 to apparent power; individual vibration orientations, engineering units and plant timezone remain unverified. Without validated operating states and maintenance context, a sensor change cannot reliably be attributed to degradation. [2, 6]",
        ),
    ],
    [
        ("h", "Measured results across the experiments"),
        (
            "p",
            "Table 2 reports February–April development results. Denominators differ with feature availability; rows are not all directly comparable. “Nominal alerts” means the fraction of publisher-selected nominal readings flagged, not an independently adjudicated operational false-alarm probability.",
        ),
        (
            "table",
            (
                ["Model or rule", "Groups warned", "Nominal alerts", "Interpretation"],
                [
                    ["Frozen LSTM native", "1/3", "1.36%", "Limited baseline warning"],
                    ["Normal-trained trends", "0/2", "0.00%", "No recovered event"],
                    ["Trend plus LSTM", "0/2", "0.09%", "No recovered event"],
                    ["Corrected supervised boosting", "0/3", "0.12%", "Misses all eligible groups"],
                    ["Ordinary IF 500 trees", "0/4", "1.48%", "Extra coverage adds no warning"],
                    ["Fully extended IF", "0/4", "1.35%", "No primary-period warning"],
                    ["Conditional vibration", "0/4", "8.86%", "Misses groups and exceeds budget"],
                    [
                        "Shape convolution single reading",
                        "3/3",
                        "2.52%",
                        "More warnings and many alarms",
                    ],
                    [
                        "Shape convolution three readings",
                        "0/3",
                        "0.12%",
                        "Persistence removes warnings",
                    ],
                ],
            ),
        ),
        (
            "p",
            "Sources: [3–7]. Trend comparisons share 4,638 endpoints and two eligible faults; LSTM also warns 0/2 on those endpoints. The supervised comparison shares 6,452 endpoints, where LSTM warns 1/3 at 1.34% nominal alerts. The forest comparisons have four native eligible groups but three on shared endpoints, with 0/3 warnings versus LSTM 1/3. The sequence comparison shares 6,469 endpoints with the native LSTM.",
        ),
        ("h", "Temporal generalization is weak"),
        (
            "p",
            "The corrected supervised classifier’s point-level ROC AUC falls from 0.902 in fitting to 0.522 in January and 0.503 in February–April. Its ranking becomes close to chance on later readings. These correlated point-level diagnostics support weak temporal transfer; they do not identify whether the cause is overfitting, changing operating conditions, label ambiguity or a combination. [3]",
        ),
        (
            "p",
            "Earlier backtests compared boosted trees and regularized logistic regression using absolute and relative-change features. None passed both the warning and alarm requirements in November and December. The only November detection among these candidates came with 4.40% nominal point alerts and did not transfer to December. Extended IF warned one November group but none in December or the primary period. Changing model family alone did not establish stable warning. [4, 6]",
        ),
    ],
    [
        ("h", "Why the promising sequence result is not yet success"),
        (
            "p",
            "Random-convolution shape features with a single-reading rule warned 2/3, 5/5 and 3/3 eligible groups in November, December and February–April. The 10/11 total is encouraging development evidence. It combines chronological refits with reused history, rather than eleven independent trials of one frozen model, and the best candidate was selected retrospectively. [7]",
        ),
        (
            "table",
            (
                ["February–April measure", "Frozen LSTM", "Shape convolution"],
                [
                    ["Eligible groups warned", "1/3", "3/3"],
                    ["Nominal readings flagged", "80/5,904", "149/5,904"],
                    ["Nominal point-alert fraction", "1.36%", "2.52%"],
                    ["Nominal alert episodes", "26", "117"],
                    ["Episodes per observed nominal day", "1.28", "5.75"],
                ],
            ),
        ),
        (
            "p",
            "Table 3. Same 6,469 score endpoints; rates use 488.75 observed nominal hours, not full calendar exposure. First warning leads for the three eligible events were 44.2, 280.1 and 75.4 minutes. These are local retrospective warnings, not guaranteed lead times. [7]",
        ),
        ("h", "The alarm burden changes the interpretation"),
        (
            "p",
            "The sequence candidate exceeds the previous 2% point-alert ceiling in all three periods. Requiring three consecutive exceedances removes every eligible warning across those periods. This shows that its useful-looking signals are brief under the tested policy. Lowering persistence gains sensitivity at the cost of more interruptions; lowering a threshold cannot be treated as a free improvement. [7]",
        ),
        ("h", "Frequent alarms can coincide with fault windows"),
        (
            "p",
            "Post-selection timing controls shifted alarms within continuous observed segments while preserving their local frequency and much of their clustering. In the primary period, 30.2% of 2,000 shifts matched or exceeded the observed 3/3 warned groups; the mean was 2.09 groups. This weakens the inference that 3/3 demonstrates predictive timing. The controls are limited by short segments and are descriptive, not a confirmatory significance test. [7]",
        ),
        ("h", "Earlier apparent improvements also need qualification"),
        (
            "p",
            "The earlier hybrid notebook used ordinary KFold with future rows in some training folds and selected thresholds on observations later reported as evaluation. Its positive results therefore do not establish future performance. The corrected supervised package restored previously omitted earlier fault examples, yet later performance remained poor. The current conclusion is not based solely on that initial packaging omission. [3, 8]",
        ),
    ],
    [
        ("h", "Conclusion and recommended next steps"),
        (
            "p",
            "The defensible conclusion is that reliable SHP early warning remains unproven because the tested models do not jointly demonstrate stable event detection, acceptable alarm burden and independent future-period performance. Small effective event counts, missing coverage and weak event semantics constrain both learning and evaluation. The measured failures are clear; their underlying physical causes remain uncertain.",
        ),
        (
            "p",
            "Retain the LSTM as the anomaly-monitoring baseline and the shape-convolution model as an experimental candidate. Do not describe either as a validated fault predictor. Anomaly detection, a warning observed before a recorded fault, and a confirmed diagnosis should remain separate claims.",
        ),
        (
            "p",
            "Before another broad model search, obtain failure-mode and maintenance records, verified event timing, units and operating-state information if available. Agree a specific actionable failure target, warning horizon and alarm budget with the mentor. The six-hour horizon should be justified by the intervention it enables, rather than by which setting gives the best retrospective score.",
        ),
        (
            "p",
            "Freeze the complete model, preprocessing, threshold, persistence, coverage and episode rules before independent evaluation. Previously inspected February–April and May-onward results cannot be renamed untouched tests. If genuinely uninspected data are unavailable, report that limitation and retain a research proof-of-concept claim. Any dataset replacement or change to approved objectives requires a separate scope decision. [9]",
        ),
        ("h", "Evidence references"),
        (
            "p",
            "[1] de Santis and Costa (2020), Extended Isolation Forests for Fault Detection in Small Hydroelectric Plants. Sustainability 12, 6421. https://doi.org/10.3390/su12166421. Source split and channels; its anomaly-detection results do not validate this project’s chronological six-hour objective.",
        ),
        (
            "p",
            "[2] docs/dataset-suitability-verdict.md and docs/results/dataset-suitability-2026-10-24/ — verified inventory, labels and coverage.",
        ),
        (
            "p",
            "[3] docs/supervised-precursor-results.md and docs/results/supervised-precursor-v1/ — corrected package, classifier and matched baseline.",
        ),
        (
            "p",
            "[4] docs/precursor-transfer-results.md and docs/results/precursor-transfer-v1/ — earlier chronological transfer comparisons.",
        ),
        ("p", "[5] docs/normal-trend-results.md — fixed trend and combined-score comparisons."),
        (
            "p",
            "[6] docs/eif-context-results.md; docs/shp-source-code-review.md; docs/results/eif-context-v1/ — forests, residual model and channel mapping.",
        ),
        (
            "p",
            "[7] docs/sequence-pattern-results.md and docs/results/sequence-patterns-v1/ — full comparisons, selection.json and timing-controls.json.",
        ),
        (
            "p",
            "[8] docs/colab-hybrid-review.md — review of saved notebook outputs, not a fresh reproduction.",
        ),
        (
            "p",
            "[9] GitHub tracking: https://github.com/timkm-dev/hydropredict/issues/19 (package), /20 (comparison) and /21 (independent confirmation). Read 1 November 2026. This report does not establish supervisor approval or close these issues.",
        ),
    ],
]


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
    sec.top_margin = sec.bottom_margin = Inches(0.65)
    sec.left_margin = sec.right_margin = Inches(0.7)
    for name in ["Normal", "Title", "Heading 1"]:
        doc.styles[name].font.name = "Calibri"
        doc.styles[name].font.color.rgb = RGBColor(0, 0, 0)
    normal = doc.styles["Normal"]
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.line_spacing = 1.06
    doc.styles["Title"].font.size = Pt(26)
    doc.styles["Heading 1"].font.size = Pt(15)
    doc.styles["Heading 1"].paragraph_format.space_before = Pt(10)
    doc.styles["Heading 1"].paragraph_format.space_after = Pt(6)
    sec.header.paragraphs[0].text = "HYDROPREDICT  |  SHP RESEARCH REVIEW"
    sec.header.paragraphs[0].runs[0].font.size = Pt(8)
    footer = sec.footer.paragraphs[0]
    footer.text = "Mentor review | 1 November 2026                                                 "
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    for run in footer.runs:
        run.font.size = Pt(8)
    md = []
    for i, blocks in enumerate(PAGES):
        if i:
            doc.add_page_break()
        for kind, content in blocks:
            if kind != "table":
                doc.add_paragraph(
                    content, style={"title": "Title", "h": "Heading 1", "p": "Normal"}[kind]
                )
                md.append(
                    ("## " if kind == "h" else "# " if kind == "title" else "") + content + "\n"
                )
                continue
            headers, rows = content
            t = doc.add_table(rows=1, cols=len(headers))
            t.autofit = False
            widths = [2.6] + [(6.87 - 2.6) / (len(headers) - 1)] * (len(headers) - 1)
            for j, h in enumerate(headers):
                t.rows[0].cells[j].text = h
            for row in rows:
                for cell, value in zip(t.add_row().cells, row, strict=False):
                    cell.text = value
            for ri, row in enumerate(t.rows):
                pr = row._tr.get_or_add_trPr()
                pr.append(OxmlElement("w:cantSplit"))
                if ri == 0:
                    pr.append(OxmlElement("w:tblHeader"))
                for j, cell in enumerate(row.cells):
                    cell.width = Inches(widths[j])
                    cp = cell._tc.get_or_add_tcPr()
                    borders = OxmlElement("w:tcBorders")
                    for edge in ["top", "left", "bottom", "right"]:
                        e = OxmlElement("w:" + edge)
                        e.set(qn("w:val"), "single")
                        e.set(qn("w:sz"), "4")
                        e.set(qn("w:color"), "D9D9D9")
                        borders.append(e)
                    cp.append(borders)
                    if ri == 0:
                        shade = OxmlElement("w:shd")
                        shade.set(qn("w:fill"), "E8EFF3")
                        cp.append(shade)
                    for p in cell.paragraphs:
                        p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(5)
                        for run in p.runs:
                            run.font.size = Pt(9.5)
                            run.bold = ri == 0
            md.extend(
                [
                    "| " + " | ".join(headers) + " |",
                    "| " + " | ".join(["---"] * len(headers)) + " |",
                ]
            )
            md.extend("| " + " | ".join(row) + " |" for row in rows)
            md.append("")
    OUT.parent.mkdir(exist_ok=True, parents=True)
    doc.save(OUT)
    OUT.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
