"""Build a mentor review DOCX from saved CARE experiment evidence."""

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path("docs/results")
OUT = Path("docs/reports/CARE_experiment_report_for_mentor.docx")


def read(folder, name="summary.json"):
    return json.loads((ROOT / folder / name).read_text())


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = sec.bottom_margin = Inches(0.7)
    sec.left_margin = sec.right_margin = Inches(0.7)
    styles = doc.styles
    for name in ["Normal", "Body Text"]:
        styles[name].font.name = "Calibri"
        styles[name].font.size = Pt(11)
        styles[name].paragraph_format.space_after = Pt(7)
        styles[name].paragraph_format.line_spacing = 1.08
    styles["Title"].font.name = "Calibri"
    styles["Title"].font.size = Pt(26)
    styles["Title"].font.color.rgb = RGBColor(0, 0, 0)
    for name, size in [("Heading 1", 17), ("Heading 2", 12)]:
        styles[name].font.name = "Calibri"
        styles[name].font.size = Pt(size)
        styles[name].font.color.rgb = RGBColor.from_string("254B60")
        styles[name].paragraph_format.space_before = Pt(10)
        styles[name].paragraph_format.space_after = Pt(6)
    header = sec.header.paragraphs[0]
    header.text = "HYDROPREDICT  |  CARE RESEARCH REVIEW"
    header.style = "Caption"
    header.runs[0].font.size = Pt(8)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("Mentor review  •  31 October 2026  |  ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    for run in footer.runs:
        run.font.size = Pt(8)

    def p(text, bold=False):
        paragraph = doc.add_paragraph()
        run = paragraph.add_run(text)
        run.bold = bold
        return paragraph

    def h(text):
        doc.add_heading(text, level=1)

    def sub(text):
        doc.add_heading(text, level=2)

    def page(title):
        doc.add_page_break()
        h(title)

    def table(headers, rows, widths):
        t = doc.add_table(rows=1, cols=len(headers))
        t.autofit = False
        for i, (label, width) in enumerate(zip(headers, widths, strict=True)):
            t.columns[i].width = Inches(width)
            t.rows[0].cells[i].text = label
        for values in rows:
            for cell, text in zip(t.add_row().cells, values, strict=True):
                cell.text = str(text)
        for ri, row in enumerate(t.rows):
            trpr = row._tr.get_or_add_trPr()
            cant = OxmlElement("w:cantSplit")
            trpr.append(cant)
            if ri == 0:
                repeat = OxmlElement("w:tblHeader")
                trpr.append(repeat)
            for i, cell in enumerate(row.cells):
                cell.width = Inches(widths[i])
                pr = cell._tc.get_or_add_tcPr()
                if ri == 0:
                    shade = OxmlElement("w:shd")
                    shade.set(qn("w:fill"), "E8EFF3")
                    pr.append(shade)
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_after = Pt(5)
                    paragraph.paragraph_format.space_before = Pt(5)
                    for run in paragraph.runs:
                        run.font.size = Pt(9.5)
                        run.bold = ri == 0
        doc.add_paragraph().paragraph_format.space_after = Pt(0)
        return t

    doc.add_paragraph("CARE predictive warning experiments", style="Title")
    p("Results and questions for mentor review", True)
    p("Prepared for the project mentor and project owner • 31 October 2026")
    p(
        "Project title: A Machine Learning–Based Early Fault Warning System for Renewable Energy Equipment"
    )
    h("Conclusion and purpose")
    p(
        "We have fitted seven machine-learning approaches on CARE. None of the tested configurations met both our early-detection and normal-alarm requirements. We seek a mentor’s review of the prediction target, evaluation design and next experiment before committing the project to this dataset.",
        True,
    )
    p(
        "This is evidence of unsuccessful configurations, not proof that CARE is unusable or that predictive modelling is impossible. Some approaches detect labelled problems but raise too many alarms; stricter approaches often miss the problems. The models and unsuccessful results have been retained."
    )
    h("What this report covers")
    p(
        "The report covers the CARE experiments recorded in this project, including the broader model-family comparison and its separate threshold-calibration check. It does not combine results from the project’s SHP or CWRU datasets."
    )
    table(
        ["Evidence scope", "Farm A", "Farm B"],
        [
            ["Files evaluated", "22", "15"],
            ["Turbines represented", "5", "9"],
            ["Anomaly cases / normal controls", "12 / 10", "6 / 9"],
            ["Normal observation-days", "181.00", "234.85"],
        ],
        [3.5, 1.6, 1.6],
    )
    p(
        "The local collection contains 95 files from 36 turbines. This modelling study used 37 files from A and B; Farm C’s model performance remains unexamined, although its metadata and data quality were inspected."
    )
    sub("Decision requested")
    p(
        "Should we revise the target or testing assumptions, undertake one further bounded study, or narrow the project to anomaly detection and maintenance decision support? No dataset or application-model replacement has been made."
    )

    page("Models and evaluation design")
    table(
        ["Approach actually fitted", "Learning task", "Main distinction"],
        [
            ["Isolation Forest", "Learn normal sensor behaviour", "Unsupervised anomaly score"],
            [
                "Extra Trees temperature model",
                "Predict expected temperatures",
                "Alarm on temperature deviations from operating conditions",
            ],
            [
                "Histogram gradient boosting",
                "Learn labelled early degradation",
                "Shallow boosted trees on component trajectories",
            ],
            [
                "Logistic regression",
                "Learn labelled early degradation",
                "Regularized linear classifier",
            ],
            [
                "Gaussian naive Bayes",
                "Learn labelled early degradation",
                "Probabilistic classifier with simplifying independence assumptions",
            ],
            [
                "Random forest classifier",
                "Learn labelled early degradation",
                "Bagged nonlinear trees",
            ],
            [
                "Approximate RBF kernel SVM",
                "Learn labelled early degradation",
                "Kernel feature approximation with a linear SVM",
            ],
        ],
        [2.05, 2.0, 2.65],
    )
    sub("How leakage was controlled")
    p(
        "The first two approaches used each file’s earlier normal history, followed by later calibration and prediction periods. For supervised approaches, all files of the evaluated turbine were excluded from classifier fitting and threshold selection. Separate inner turbine exclusions produced normal-control calibration scores. Adaptation to the evaluated turbine’s earlier normal history was still allowed."
    )
    p(
        "Status codes, timestamps and identifiers were not predictive inputs. Status labels were used only to select historical normal observations where specified. Temporal features use past and current readings and reset at gaps. Anomalous examples used for classifier fitting exclude the last 24 hours of their annotated interval."
    )
    sub("What is and is not comparable")
    p(
        "The four new classifiers share the same residual trajectories, sampled examples, weights and turbine splits as the supervised experiment. Their comparison is controlled. The earlier normal-only methods solve a different learning task and are contextual references rather than pure architecture substitutions."
    )
    p(
        "This is development evidence on previously inspected A/B data, not a final independent test or a globally chronological deployment trial. Repeated files from a turbine are correlated; millions of sensor readings do not constitute millions of independent fault examples."
    )

    page("Results under the original alarm rules")
    p(
        "An early detection means at least one persistent alert inside the labelled anomaly interval and at least 24 hours before its annotated end. This is not verified time before physical failure. Six consecutive abnormal 10-minute readings are required; gaps reset persistence."
    )
    p(
        "Our provisional screening requirements were at least 50% early detection, no more than one normal alarm episode per 30 observed days, no more than 1% of normal rows alerted, and at least 80% valid prediction coverage in every file. These are engineering assumptions requiring mentor or supervisor review."
    )
    series = []
    for folder, label in [
        ("care-baseline-v1", "Isolation Forest"),
        ("care-residual-v2", "Extra Trees residuals"),
        ("care-supervised-v3", "Gradient boosting"),
    ]:
        series.append((label, {r["farm"]: r for r in read(folder)}))
    added = read("care-model-comparison-v4")
    for family, label in [
        ("logistic", "Logistic regression"),
        ("gaussian_nb", "Gaussian naive Bayes"),
        ("random_forest", "Random forest"),
        ("rbf_svm_approx", "Approx RBF SVM"),
    ]:
        series.append((label, {r["farm"]: r for r in added if r["family"] == family}))
    rows = []
    for label, data in series:
        a, b = data["Wind Farm A"], data["Wind Farm B"]
        assert not a["gate_pass"] and not b["gate_pass"]
        rows.append(
            [
                label,
                f"{a['detected_24h']}/12\n{12 - a['detected_24h']} missed",
                f"{b['detected_24h']}/6\n{6 - b['detected_24h']} missed",
                f"{a['normal_episodes_per_30_days']:.2f}\n{100 * a['normal_alert_fraction']:.2f}%",
                f"{b['normal_episodes_per_30_days']:.2f}\n{100 * b['normal_alert_fraction']:.2f}%",
                "No",
            ]
        )
    table(
        [
            "Model",
            "A early",
            "B early",
            "A alarms\nrate / rows",
            "B alarms\nrate / rows",
            "Pass both farms",
        ],
        rows,
        [1.75, 0.85, 0.85, 1.3, 1.3, 0.65],
    )
    p(
        "In the alarm columns, the first number is episodes per 30 observed normal days; the second is the percentage of normal readings in an alert state. “Missed” means not detected under the 24-hour rule, including alerts that arrived too late."
    )
    sub("What the comparison shows")
    p(
        "Extra Trees residual monitoring detected the most early cases in A and tied Isolation Forest in B, but its normal alarm rates were too high. The later supervised approaches mostly reduced alarms by becoming quiet, without retaining useful detection. Logistic regression detected one early case in each farm but still failed the full requirements."
    )
    p(
        "A no-alarm reference would detect zero cases and generate zero alarms. Quiet behaviour alone is therefore not success. No model was selected or deployed from this table."
    )

    page("Checking whether score calibration changes the answer")
    p(
        "Different models produce scores on different scales. We preserved the original run and performed a separate calibration check using each model’s native scores and normal-control quantiles. Outer classifiers were reused unchanged; only the predeclared calibration procedure changed. Evaluated fault cases were not used to select thresholds."
    )
    native = read("care-model-calibration-v4")
    rows = []
    names = {
        "logistic": "Logistic regression",
        "gaussian_nb": "Gaussian naive Bayes",
        "random_forest": "Random forest",
        "rbf_svm_approx": "Approx RBF SVM",
    }
    for family in names:
        for farm, den in [("Wind Farm A", 12), ("Wind Farm B", 6)]:
            before = next(r for r in added if r["family"] == family and r["farm"] == farm)
            after = next(r for r in native if r["family"] == family and r["farm"] == farm)
            assert not after["gate_pass"]
            rows.append(
                [
                    names[family],
                    farm[-1],
                    f"{before['detected_24h']} → {after['detected_24h']} / {den}",
                    f"{before['normal_episodes_per_30_days']:.2f} → {after['normal_episodes_per_30_days']:.2f}",
                    f"{100 * after['normal_alert_fraction']:.2f}%",
                    "No",
                ]
            )
    table(
        [
            "Classifier",
            "Farm",
            "Early cases\noriginal → revised",
            "Normal episodes\noriginal → revised",
            "Revised normal rows",
            "Revised pass",
        ],
        rows,
        [1.6, 0.45, 1.35, 1.45, 1.1, 0.75],
    )
    p(
        "Episode rates remain per 30 observed normal days. The revised calibration did not make any classifier pass. It recovered one early case for the approximate SVM in B. Gaussian naive Bayes detected four A cases within their intervals, but all were too late for the 24-hour requirement; early detection remains zero."
    )
    sub("Why both tables are retained")
    p(
        "The original fixed score grid can suppress models whose outputs saturate or use different scales. This sensitivity check addresses that limitation without silently replacing an unfavourable result. It is another configuration on exposed development data, not independent confirmation or four new model families."
    )
    sub("Candidates not yet tested on CARE")
    p(
        "Autoencoders, LSTM or GRU networks, temporal convolutional networks, exact kernel SVC, and survival or time-to-failure models have not been fitted in this CARE study. Neural approaches would need a defensible design for the small number of independent fault cases. Survival methods would require verified failure and censoring definitions. We have not exhausted all possible models."
    )

    page("Three examples to demonstrate to the mentor")
    p(
        "All three examples use the frozen Extra Trees residual monitor on Farm B. They were selected to illustrate different outcomes, not to estimate overall performance. Blue shows score divided by threshold; the dashed red line is the threshold; red points are persistent alerts. Shading marks the labelled anomaly interval and the dotted line marks the 24-hour cutoff."
    )
    captions = {
        19: "Event 19 was detected. The first in-interval alert occurred 331.7 hours before the annotated end. This is an end-relative measure, not a verified physical-failure lead time.",
        7: "Event 7 was missed. It has the same broad description as event 19, high temperature in the transformer cell, but produced no persistent alert. A successful example did not generalize to every similar label.",
        2: "Event 2 is labelled normal. The model produced four alarm episodes and alerted on 204 of 2,215 readings. These are unnecessary alarms against the benchmark labels, not a verified fleet-wide false-alarm rate.",
    }
    for event_id in [19, 7, 2]:
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.add_run().add_picture(
            f".local/care-mentor-report/case-{event_id}.png", width=Inches(6.7)
        )
        paragraph.paragraph_format.keep_with_next = True
        caption = p(captions[event_id])
        caption.paragraph_format.space_after = Pt(6)
        for run in caption.runs:
            run.font.size = Pt(10)

    page("Interpretation and limits of the evidence")
    sub("Fitting is not the same as reliable warning")
    p(
        "The supervised gradient-boosting model had fitting discrimination scores of 0.908–0.945 in A and 0.953–0.975 in B. On excluded turbines with both classes, those scores fell to 0.314–0.547 and 0.047–0.668. These are AUC values, not accuracy percentages; a value near 0.5 indicates little ranking discrimination."
    )
    p(
        "The broader comparison shows a similar problem. Random forest’s median fitting AUC was about 0.986 in A and 0.989 in B, compared with 0.475 and 0.556 on excluded turbines. Logistic regression transferred somewhat better by this ranking measure, but still did not produce an acceptable warning system at the alarm limits."
    )
    sub("What we can conclude")
    p(
        "We can fit several model families and reproduce their predictions. None of the tested configurations simultaneously achieved the provisional detection and alarm criteria. The supervised trajectory representation did not reliably transfer to excluded turbines. These findings justify pausing a dataset pivot pending expert review."
    )
    sub("What we cannot conclude")
    p(
        "We cannot conclude that CARE is intrinsically unusable, that all predictive-maintenance methods fail, or that Farm C will behave like A and B. The newer classifiers share one temperature-residual representation; changing their architecture does not test every possible sensor feature set. Fault types, machine differences, operating changes and label uncertainty remain plausible explanations, not established causes."
    )
    p(
        "The data are curated event files. Normal observation-days are available rows divided by 144, not a continuously observed real fleet. Fault-derived status labels in A cannot independently verify operational downtime. Exact physical failure timing, practical actionability and the local archive release remain incompletely verified."
    )
    sub("Uncertainty and repeated experiments")
    p(
        "Only five A and nine B turbines are represented in modelling. A previous 2,000-replicate turbine-cluster bootstrap gave the supervised B normal-alarm rate an empirical range of 0–1.64 episodes per 30 days. Its zero-detection bootstrap range is degenerate because every observed detection was zero; it does not prove zero population detection or zero uncertainty. Repeated development experiments further limit claims of confirmation."
    )

    page("Questions and evidence for the review meeting")
    table(
        ["Question for the mentor", "Why it changes the next step"],
        [
            [
                "Is the target early degradation detection or prediction of a precisely timed failure?",
                "The available annotations support an early-degradation proxy; exact failure timing needs clarification.",
            ],
            [
                "Are 24 hours and one normal alarm per 30 days appropriate?",
                "These provisional assumptions determine what counts as useful. They are not supervisor-approved operational requirements.",
            ],
            [
                "Should success require transfer across turbines?",
                "A machine-specific demonstration and an unseen-turbine system are different objectives. The test must match the approved scope.",
            ],
            [
                "Should we focus on one fault type or broaden the sensor representation?",
                "Pooling different mechanisms may obscure transferable patterns; more complexity alone has not solved the problem.",
            ],
            [
                "What would count as independent confirmation?",
                "A/B are development-exposed. Farm C has uninspected performance but would need an agreed protocol before evaluation.",
            ],
        ],
        [3.2, 3.5],
    )
    p(
        "Proposed next decision: agree one revised target and evaluation plan before additional training, or formally narrow the project to anomaly detection and maintenance support. Retain the failed experiments as part of the evidence rather than deleting or rebranding them."
    )
    sub("Audit trail")
    p(
        "The repository contains frozen protocols, per-case results, model and source hashes, calibration audits, scripts and saved model artifacts. The four new families completed fitting without recorded warnings. Twelve validation tests passed, including future-data isolation, whole-turbine exclusions, gap handling and equivalent alarm bookkeeping. Faster bookkeeping reproduced all 37 saved supervised case outcomes."
    )
    p(
        "Main evidence folders under docs/results: care-baseline-v1; care-residual-v2; care-supervised-v3; care-model-comparison-v4; care-model-calibration-v4. Models and timestamp-level scores are kept locally under ignored .local folders. This document reports existing outputs; no model was retrained to prepare it."
    )
    p(
        "Protocols: docs/care-baseline-protocol.md; care-residual-protocol.md; care-supervised-protocol.md; care-model-comparison-protocol.md; care-model-calibration-addendum.md. Project tracking: github.com/timkm-dev/hydropredict/issues/21."
    )
    sub("Dataset references")
    p(
        "Gück, Roelofs and Faulstich, CARE to Compare, Data 2024, 9(12), 138. DOI: 10.3390/data9120138. Publisher release notes: https://zenodo.org/records/15846963. Label interpretation: https://aefdi.github.io/EnergyFaultDetector/care2compare_faq.html. These sources describe the benchmark; they do not validate our models."
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(str(OUT.resolve()))


if __name__ == "__main__":
    main()
