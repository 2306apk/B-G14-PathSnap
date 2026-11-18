"""Reproducible IMPLEMENTATION GUIDE, not a substituted team-authored final report.

Every numerical result is loaded from measured CSV/JSON files, never hardcoded.
Report screenshots, experimental explanations, and authorship still belong to team.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether,Preformatted

DARK=colors.HexColor('#17233b');BLUE=colors.HexColor('#2563eb');GRAY=colors.HexColor('#64748b')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',default='results/full')
    ap.add_argument('--out',default='docs/IMPLEMENTATION_GUIDE.pdf')
    a=ap.parse_args(); root=Path(a.results);output=Path(a.out);output.parent.mkdir(parents=True,exist_ok=True)
    models=pd.read_csv(root/'model_comparison.csv')
    tune=json.loads((root/'tuning_summary.json').read_text())
    decoded=pd.read_csv(root/'decoder_comparison.csv')
    inv=pd.read_csv(root/'inventory.csv')
    ss=getSampleStyleSheet()
    ss.add(ParagraphStyle(name='BigTitle2',parent=ss['Title'],fontName='Helvetica-Bold',fontSize=25,leading=31,textColor=DARK,spaceAfter=18))
    ss.add(ParagraphStyle(name='Section2',parent=ss['Heading2'],fontSize=13,leading=17,textColor=BLUE,spaceBefore=13,spaceAfter=7))
    ss.add(ParagraphStyle(name='Small2',parent=ss['BodyText'],fontSize=9,leading=13))
    ss.add(ParagraphStyle(name='Code2',fontName='Courier',fontSize=8,leading=11,textColor=DARK,backColor=colors.HexColor('#f4f7fc'),leftIndent=10))
    story=[]
    def title(x):story.append(Paragraph(x,ss['BigTitle2']))
    def head(x):story.append(Paragraph(x,ss['Section2']))
    def para(x):story.append(Paragraph(x,ss['Small2']));story.append(Spacer(1,6))
    def tab(headings,rows,widths=None):
        data=[[Paragraph(f'<font color="white"><b>{h}</b></font>',ss['Small2']) for h in headings]]+[[Paragraph(str(x),ss['Small2']) for x in row] for row in rows]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),DARK),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f6fd')]),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('BOTTOMPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),
        ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#cbd5e1'))]))
        story.append(t);story.append(Spacer(1,10))
    def plot(file,width=16*cm):
        f=root/file
        if f.is_file():
            from PIL import Image as PILImage
            with PILImage.open(f) as im:w,h=im.size
            story.append(Image(str(f),width=width,height=width*h/w,hAlign='CENTER'))
            story.append(Spacer(1,8))
    def page(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(GRAY)
        canvas.drawString(42,34,'PathSnap | Measured implementation guide (not final team submission)')
        canvas.drawRightString(A4[0]-42,34,f'Page {doc.page}')
    title('PathSnap | Implementation Guide')
    para('<b>Distributed Map Matching and Route Reconstruction</b><br/>PES University | UE24CS352A Machine Learning')
    para('<font color="#b45309"><b>STATUS: AUTOMATED ENGINEERING HANDOFF, NOT THE TEAM\'S FINAL ACADEMIC REPORT.</b> The team must independently reproduce results, review the code, capture its own screenshots and add original experimental interpretation, consistent with the organiser\'s integrity policy.</font>')
    head('What was inspected')
    train=inv[inv.partition=='train'];test=inv[inv.partition=='student_test']
    tab(['Partition','Trip files','GPS observations','Road arcs'],[
        ['Training',len(train),f'{train.observations.sum():,}',f'{train.road_arcs.sum():,}'],
        ['Student-testing',len(test),f'{test.observations.sum():,}',f'{test.road_arcs.sum():,}'],
        ['Evaluator-testing','10 expected','Not provided','Not provided']],
        [3.2*cm,3*cm,4.2*cm,4.3*cm])
    para('Each trip supplies a longitude/latitude/time <b>.track</b>, longitude/latitude <b>.nodes</b>, directed node-to-node <b>.arcs</b>, and an ordered list of arc IDs in <b>.route</b>. Importantly, the route is <b>not</b> a pointwise GPS label table.')
    head('Pipeline')
    para('Raw GPS and graph → local metre projection → candidate directed arcs → fourteen candidate features → five classifier experiments → grouped trip cross-validation → Optuna TPE tuning → candidate probabilities → road-network HMM/Viterbi → predicted arc IDs → FastAPI on Laptop 1 → separate browser UI on Laptop 2.')
    head('Integrity and evaluation boundary')
    para('Reference routes are used to infer approximate labels for allowed train/student-test experiments, and only <b>after</b> inference for aggregate server-side scoring. Predictions and evaluator-facing UI are not given reference arc IDs, pointwise truth or a copied reference polyline. The reserved evaluator partition is not supplied in the two original archives.')
    story.append(PageBreak())
    title('EDA and geographical data')
    head('Actual dataset distributions')
    from PIL import Image as PILImage
    hist_images=[]
    for filename in ('eda/gps_counts.png','eda/time_gaps.png'):
        path=root/filename
        with PILImage.open(path) as img:iw,ih=img.size
        hist_images.append(Image(str(path),width=7.3*cm,height=7.3*cm*ih/iw))
    story.append(Table([hist_images],colWidths=[7.6*cm,7.6*cm],hAlign='CENTER'))
    story.append(Spacer(1,8))
    para('GPS is nominally sampled near 1 Hz but contains irregular gaps; preprocessing validates monotonic timestamps, coordinate range, node IDs and arc indices. The local projection uses longitude metres scaled by cosine(latitude), rather than treating degrees as distances.')
    head('A source-backed road network plot')
    plot('eda/trip_gps_roads.png',width=12.5*cm)
    para('The interactive Folium HTML (`results/full/folium_trip_00000052.html`) contains real GPS, predicted arcs, nearby roads and the student-test reference route. For the required final report, the team must capture a genuine Folium browser screenshot and insert it, rather than fabricate one.')
    story.append(PageBreak())
    title('Five-model experiment')
    para('All 80 training trips were used. Group-aware three-fold cross-validation avoids sharing a trip between training and validation folds. For runtime practicality, train sampling stride = 12; student-testing stride = 1 across all ten available student trips. Eight candidate arcs were generated per observation. Selection metric = training CV Top-1 accuracy.')
    rows=[]
    for _,m in models.iterrows():
        rows.append([m['model'].replace('_',' '),f'{m.cv_top1_accuracy*100:.2f}%',f'{m.student_top1_accuracy*100:.2f}%',f'{m.student_candidate_macro_f1:.3f}',f'{m.student_candidate_log_loss:.3f}'])
    tab(['Model','CV Top-1','Student Top-1','Cand. Macro F1','Cand. log loss'],rows,[4.1*cm,2.5*cm,3.1*cm,3*cm,3.2*cm])
    plot('student_top1.png',width=15.2*cm)
    para('Reported Macro F1 is <b>binary candidate classification Macro F1</b>, not macro-averaged over every unique trip-local road arc class. Student-testing Top-1 is based on geometrically aligned reference-route <i>proxy labels</i>; interpret it accordingly.')
    story.append(PageBreak())
    title('Tuning and HMM evaluation')
    head('Bayesian hyperparameter tuning')
    baseline=float(tune['baseline_cv_accuracy']);best=float(tune['best_cv_accuracy'])
    p0=float(tune['baseline_student_test']['observation_top1_accuracy'])
    p1=float(tune['tuned_student_test']['observation_top1_accuracy'])
    tab(['TPE search evidence','Measured result'],[
        ['Selected model',tune['model']],['Completed Optuna trials',tune['trials']],
        ['Baseline CV Top-1',f'{100*baseline:.2f}%'],['Tuned CV Top-1',f'{100*best:.2f}%'],
        ['Baseline student Top-1',f'{100*p0:.2f}%'],['Tuned student Top-1',f'{100*p1:.2f}%']],[6.7*cm,7.5*cm])
    para('<b>Best params (actual):</b> '+', '.join(f'{k}={v}' for k,v in tune['best_params'].items()))
    para('Tuning objective: maximise training-only trip-grouped CV Top-1 accuracy. The separate student-testing partition is measured only after the trial search. Search history and configurations are in `results/full/optuna_trials.csv` and the SQLite Optuna study.')
    head('Classifier-only versus HMM / Viterbi')
    def wt(k):return 100*float((decoded[k]*decoded.observations).sum()/decoded.observations.sum())
    tab(['Metric (weighted over all 19,160 GPS samples)','Classifier','Classifier + HMM'],[
        ['Approx. observation Top-1',f'{wt("observation_top1_accuracy"):.2f}%',f'{wt("hmm_observation_accuracy"):.2f}%'],
        ['Direct same-or-adjacent arc continuity',f'{wt("classifier_direct_continuity"):.2f}%',f'{wt("hmm_direct_continuity"):.2f}%']],
        [7.0*cm,3.5*cm,4.0*cm])
    para('Each HMM state is a candidate directed road arc; emissions are normalised model probabilities; transitions use bounded directed Dijkstra paths and travel plausibility. Viterbi finds a maximum log-probability path. Large timestamp gaps reset decoding, while invalid/disconnected transitions receive a reset penalty.')
    para('Direct continuity counts consecutive identical arcs or arcs sharing a directed endpoint. It can penalise valid skipped arcs during sparse GPS sampling; compare it alongside GPS-point accuracy and actual mapped geometry.')
    story.append(PageBreak())
    title('Linux runbook')
    head('Set up the repository')
    for cmd in [
        'unzip -q ~/Downloads/PathSnap_Implementation.zip -d ~\ncd ~/PathSnap_Implementation\n# Unzip Student folder.zip and copy train/ and student_test/ into data/',
        'python3 -m venv .venv\nsource .venv/bin/activate\npython -m pip install -r requirements.txt\npython -m pytest -q',
        'python -m scripts.inventory --data-root data --out results/inventory.csv',
        'python -m scripts.train_models --data-root data --train-stride 12 \\\n  --test-stride 1 --out-dir results/full --artifact-dir artifacts/full',
        'python -m scripts.tune --data-root data --trials 15 \\\n  --train-stride 12 --test-stride 1 --out-dir results/full \\\n  --artifact-dir artifacts/full',
        'python -m scripts.evaluate --data-root data \\\n  --model artifacts/full/best_tuned.joblib --out results/full/decoder_comparison.csv',
    ]:
        story.append(Preformatted(cmd,ss['Code2']));story.append(Spacer(1,10))
    head('Two laptops')
    story.append(Preformatted('''# Laptop 1
export PATHSNAP_DATA_ROOT="$(pwd)/data"
export PATHSNAP_MODEL_PATH="$(pwd)/artifacts/full/best_tuned.joblib"
uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000 --access-log

# Laptop 2 (UI only)
python3 -m http.server 5500 --directory ui --bind 0.0.0.0
# Open http://127.0.0.1:5500 on Laptop 2.
# Set API field to http://LAPTOP_1_IP:8000''',ss['Code2']))
    story.append(PageBreak())
    title('Team split and submission')
    tab(['Member','Workstream','Evidence for viva'],[
        ['1','Dataset, EDA, geographic projection, reference alignment','Inventory, GPS gaps, candidate-miss analysis, Folium'],
        ['2','ML features, five models, grouped CV and metrics','Training runs, model configurations, plotted comparisons'],
        ['3','Bayesian TPE, HMM transitions, Viterbi decoding','Trial database, decoder correctness, continuity/accuracy'],
        ['4','API, UI, Linux network deployment, reporting','Two-laptop requests, client screen, GitHub/report integration']],
        [1.8*cm,6.6*cm,6.4*cm])
    head('Submission gate')
    para('The organiser requires exactly two team deliverables: <b>(1) a public GitHub repository</b> containing all code, models, metrics, configs, and Folium work; <b>(2) one consolidated technical-report PDF</b> documenting original team decisions and every actual model experiment.')
    para('Before submitting, reproduce all measurements locally, analyse where HMM failed or helps, capture a Folium screenshot and real UI/API screenshots, test the two-laptop LAN, perform organiser-controlled reserved evaluation without tuning on its labels, and document each member\'s original contributions.')
    head('Files to review')
    para('`README.md` — all commands and assumptions<br/>`docs/TEAM_PLAN.md` — branch ownership and verification<br/>`docs/REPORT_CHECKLIST.md` — required final team report sections<br/>`results/full/` — real development metrics and plots<br/>`artifacts/full/` — fitted classifiers and selected tuned classifier.')
    para('<font color="#b45309"><b>Integrity reminder:</b> This PDF and the implementation are technical assistance. They must not be claimed as original independent student work without disclosure. Any final report must reflect experiments the team actually ran and understood. Do not fabricate metrics, plots, screenshots or evaluator results.</font>')
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=40,leftMargin=40,topMargin=40,bottomMargin=54,title='PathSnap Implementation Guide',author='PathSnap implementation assistance')
    doc.build(story,onFirstPage=page,onLaterPages=page)
    print('Created',output)

if __name__=='__main__':main()
