from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import altair as alt
import streamlit.components.v1 as components

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="가새 구조응답 분석", page_icon="🏗️", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container {max-width: 800px; padding-top: 4rem; padding-bottom: 3rem}
h1,h2,h3 {color:#0B294B !important; line-height:1.35 !important}
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
 width:100%; min-height:3.3rem; border-radius:12px; font-weight:650;
 white-space:normal; text-align:center; border-color:#C7D6E5;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {border-color:#2468A0; color:#2468A0}
[data-testid="stMetric"] {background:#F2F5F9; border:1px solid #DCE3EB; border-radius:12px; padding:16px}
[data-testid="stMetricLabel"] *, [data-testid="stMetricLabel"] {color:#334155 !important}
[data-testid="stMetricValue"] *, [data-testid="stMetricValue"] {color:#0B294B !important}
.hero {background:#0B294B; padding:25px 22px; border-radius:16px; margin-bottom:18px}
.hero-title {color:white; font-size:1.65rem; font-weight:800; line-height:1.35}
.hero-sub {color:#D0E2F0; font-size:.92rem; margin-top:10px}
.section-tag {border-left:5px solid #D97716; padding:7px 12px; background:#EAF1F8;
 color:#0B294B; font-weight:700; border-radius:4px; margin:12px 0}
@media (max-width:640px) {
 .block-container {padding:2rem .9rem 2.5rem}
 h1 {font-size:1.85rem !important} h2 {font-size:1.45rem !important} h3 {font-size:1.2rem !important}
 .hero {padding:20px 16px} .hero-title {font-size:1.4rem}
 [data-testid="stMetricValue"] {font-size:1.65rem !important}
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    return {name: joblib.load(BASE / filename) for name, filename in {
        "period": "period_model.pkl", "shear": "shear_model.pkl",
        "drift": "drift_model.pkl", "disp": "disp_model.pkl"
    }.items()}

@st.cache_data(show_spinner=False, max_entries=8)
def render_pdf_page(pdf_bytes, page_num):
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(1.7, 1.7), alpha=False)
    result = pix.tobytes("png")
    doc.close()
    return result

if "page" not in st.session_state:
    st.session_state.page = "home"

def go(page):
    st.session_state.page = page
    st.rerun()

def back():
    if st.button("← 홈으로 돌아가기"):
        go("home")

if st.session_state.page == "home":
    st.markdown('<div class="hero"><div class="hero-title">가새 구조응답 분석 시스템</div>'
                '<div class="hero-sub">저층 장스팬 구조물 · MIDAS Gen 구조해석 · AI 예측</div></div>',
                unsafe_allow_html=True)
    st.write("가새 형태별 해석 결과를 확인하거나 X형 가새의 높이와 폭에 따른 구조응답을 예측할 수 있습니다.")
    st.divider()
    if st.button("① 가새 형태별 구조해석 결과", key="home_shape"):
        go("shape")
    if st.button("② X형 가새 크기에 따른 AI 예측", key="home_predict"):
        go("predict")
    if st.button("③ 논문 원본", key="home_paper"):
        go("paper")
    if st.button("④ 모델 사진", key="home_photos"):
        go("photos")
    st.divider()
    st.caption("본 서비스는 연구용이며, 실제 구조설계의 적합성 판단을 대신하지 않습니다.")

elif st.session_state.page == "shape":
    st.header("가새 형태별 구조해석 결과")
    st.caption("MIDAS Gen 해석 결과 | 학사논문 제3장")
    st.info("실제 구조해석 결과이며 AI 예측값이 아닙니다.")
    order = ["무가새", "X형", "K형", "역V형"]
    st.subheader("① 가새 형태별 고유주기 비교")
    period_df = pd.DataFrame({"가새 형태":order, "고유주기 (sec)":[0.6532,0.5157,0.5233,0.5231]})
    chart = (alt.Chart(period_df).mark_bar(cornerRadiusEnd=4)
             .encode(x=alt.X("고유주기 (sec):Q", title="고유주기 (sec)", scale=alt.Scale(domain=[0,0.7])),
                     y=alt.Y("가새 형태:N", sort=order, title=None),
                     color=alt.Color("가새 형태:N", scale=alt.Scale(domain=order,
                         range=["#8EA7BD","#D97716","#2468A0","#5B86AA"]), legend=None),
                     tooltip=["가새 형태:N",alt.Tooltip("고유주기 (sec):Q",format=".4f")])
             .properties(height=245))
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(period_df, use_container_width=True, hide_index=True)
    st.success("X형 가새의 고유주기는 0.5157초로 가장 짧으며, 무가새 대비 약 21.05% 감소했습니다.")
    st.subheader("② 최대 층간변위비 비교")
    drift_df = pd.DataFrame({"가새 형태":order,"X방향":[0.0073,0.0074,0.0074,0.0074],
                             "Y방향":[0.0051,0.0046,0.0047,0.0047]})
    st.dataframe(drift_df.style.format({"X방향":"{:.4f}","Y방향":"{:.4f}"}),
                 use_container_width=True, hide_index=True)
    st.subheader("③ 최대변위 비교")
    disp_df = pd.DataFrame({"가새 형태":order,"X방향 (mm)":[13.8,13.9,13.9,13.8],
                            "Y방향 (mm)":[5.3,5.3,5.3,5.3]})
    st.dataframe(disp_df, use_container_width=True, hide_index=True)
    st.subheader("④ 밑면전단력 비교")
    shear_df = pd.DataFrame({"가새 형태":order,
        "X방향 (kN)":[2458.34,2462.275596,2463.417778,2461.678600],
        "Y방향 (kN)":[2458.34,2462.275603,2463.417786,2461.678608]})
    st.dataframe(shear_df.style.format({"X방향 (kN)":"{:.3f}","Y방향 (kN)":"{:.3f}"}),
                 use_container_width=True, hide_index=True)
    st.divider()
    st.subheader("⑤ 종합 분석 및 최적 가새 선정")
    st.markdown('<div class="section-tag">최적 가새 형태 · X형 가새</div>', unsafe_allow_html=True)
    st.write("X형 가새는 세 가지 가새 형태 중 고유주기가 가장 짧고 Y방향 층간변위비도 상대적으로 작아 본 연구의 최적 형태로 선정했습니다. 다만 최대변위와 밑면전단력의 차이는 크지 않으므로 모든 구조응답에서 절대적으로 우수하다는 뜻은 아닙니다.")
    st.caption("출처: 학사학위논문 표 3.1~3.4")
    back()

elif st.session_state.page == "predict":
    st.header("X형 가새 크기에 따른 AI 예측")
    st.caption("가새 형태: X형 고정 | 입력: 높이 A, 폭 B (m)")
    st.warning("11개 크기 조건으로 학습한 연구용 추정값입니다. 입력 범위 내의 미학습 조합도 검증된 값은 아닙니다.")
    with st.form("prediction"):
        height = st.number_input("가새 높이 A (m)",min_value=4.0,max_value=7.0,value=5.5,step=0.25,format="%.2f")
        width = st.number_input("가새 폭 B (m)",min_value=4.0,max_value=6.25,value=4.25,step=0.25,format="%.2f")
        submitted = st.form_submit_button("구조응답 예측하기",use_container_width=True)
    if submitted:
        try:
            models = load_models()
            inputs = pd.DataFrame({"가새형태":[0],"size_h":[height],"size_b":[width]})
            results = {name:float(model.predict(inputs)[0]) for name,model in models.items()}
            st.subheader("AI 예측 결과")
            st.metric("고유주기",f"{results['period']:.4f} sec")
            st.metric("최대밑면전단력",f"{results['shear']:,.2f} kN")
            st.metric("최대층간변위비",f"{results['drift']:.6f}")
            st.metric("최대변위",f"{results['disp']*1000:.3f} mm")
            st.caption("최대변위는 모델 출력 단위(m)를 mm로 환산했습니다.")
        except FileNotFoundError:
            st.error("모델 파일 4개가 저장소 최상위 폴더에 있는지 확인해주세요.")
        except Exception as e:
            st.error(f"모델 실행 오류: {e}")
    back()

elif st.session_state.page == "paper":
    st.header("논문 원본")
    # 저장소에 있는 실제 PDF 파일명을 자동 확인합니다.
    pdf_names = ["paper.pdf.pdf", "pdf.pdf", "paper.pdf"]
    pdf_path = next((BASE / name for name in pdf_names if (BASE / name).is_file()), None)
    if pdf_path is None:
        st.warning("논문 PDF 파일을 찾을 수 없습니다. 저장소의 PDF 파일명을 확인해주세요.")
    else:
        pdf_bytes = pdf_path.read_bytes()
        st.download_button("📄 논문 PDF 다운로드",data=pdf_bytes,file_name="brace_thesis.pdf",
                           mime="application/pdf",use_container_width=True)
        mode = st.radio("열람 방식",["화면 너비에 맞춰 보기 (모바일 추천)","원본 PDF 뷰어"],horizontal=False)
        if mode.startswith("화면"):
            try:
                import fitz
                with fitz.open(stream=pdf_bytes,filetype="pdf") as doc:
                    total = len(doc)
                if "pdf_page" not in st.session_state:
                    st.session_state.pdf_page = 1
                st.session_state.pdf_page = max(1,min(st.session_state.pdf_page,total))
                selected = st.number_input("페이지",min_value=1,max_value=total,step=1,key="pdf_page")
                st.image(render_pdf_page(pdf_bytes,selected),use_container_width=True)
                st.caption(f"{selected} / {total} 페이지 · 페이지 전체를 화면 너비에 맞춰 표시합니다. 글자가 작으면 PDF를 다운로드해 확대해서 읽으세요.")
                left,right = st.columns(2)
                with left:
                    if st.button("← 이전 페이지",disabled=selected<=1):
                        st.session_state.pdf_page -= 1
                        st.rerun()
                with right:
                    if st.button("다음 페이지 →",disabled=selected>=total):
                        st.session_state.pdf_page += 1
                        st.rerun()
            except Exception as e:
                st.error(f"PDF 페이지 표시 오류: {e}")
                st.info("위의 PDF 다운로드 버튼을 이용해 별도 뷰어에서 열람할 수 있습니다.")
       
        else:
            import base64

            pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

            html = """
<div id="pdf-container" style="height:100%; overflow-y:auto; background:#e8edf2; padding:8px;"></div>
<script type="module">
import * as pdfjsLib from "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.min.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.worker.min.mjs";

const base64 = "PDF_BASE64";
const bytes = Uint8Array.from(atob(base64), c => c.charCodeAt(0));
const pdf = await pdfjsLib.getDocument({data: bytes}).promise;
const container = document.getElementById("pdf-container");

for (let i = 1; i <= pdf.numPages; i++) {
  const page = await pdf.getPage(i);
  const viewport = page.getViewport({scale: 1.5});

  const canvas = document.createElement("canvas");
  canvas.style.width = "100%";
  canvas.style.height = "auto";
  canvas.style.display = "block";
  canvas.style.marginBottom = "12px";
  canvas.width = viewport.width;
  canvas.height = viewport.height;

  container.appendChild(canvas);

  await page.render({
    canvasContext: canvas.getContext("2d"),
    viewport: viewport
  }).promise;
}
</script>
""".replace("PDF_BASE64", pdf_base64)

            components.html(html, height=750, scrolling=False)

            st.caption("논문을 위아래로 스크롤하여 연속 열람할 수 있습니다.")

    back()

    back()

elif st.session_state.page == "photos":
    st.header("구조해석 모델 사진")
    st.write("본 연구에서 MIDAS Gen으로 모델링한 저층 장스팬 구조물의 3차원 구조해석 모델입니다.")
    photos = [("model_1.png","그림 1. X형 가새 구조해석 모델"),
              ("model_2.png","그림 2. K형 가새 구조해석 모델"),
              ("model_3.png","그림 3. 역V형 가새 구조해석 모델")]
    for filename,caption in photos:
        image_path = BASE / filename
        if image_path.exists():
            st.image(str(image_path),caption=caption,use_container_width=True)
        else:
            st.warning(f"{filename} 파일을 찾을 수 없습니다.")
    back()
