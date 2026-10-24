from pathlib import Path
import html
import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="가새 구조응답 분석", page_icon="🏗️", layout="centered")
st.markdown("""<style>
.block-container {max-width:760px;padding-top:1.5rem;padding-bottom:3rem}
.stButton>button {width:100%;min-height:3.4rem;border-radius:12px;font-weight:650}
h1,h2,h3 {color:#0B2545}
[data-testid="stMetric"] {background:#F2F5F9;border-radius:12px;padding:14px}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    return {name: joblib.load(BASE / filename) for name, filename in {
        "period": "period_model.pkl", "shear": "shear_model.pkl",
        "drift": "drift_model.pkl", "disp": "disp_model.pkl"
    }.items()}

if "page" not in st.session_state:
    st.session_state.page = "home"

def go(page):
    st.session_state.page = page
    st.rerun()

def back():
    if st.button("← 홈으로 돌아가기"):
        go("home")

if st.session_state.page == "home":
    st.title("가새 구조응답 분석 시스템")
    st.caption("저층 장스팬 구조물 | 구조해석 결과 및 AI 기반 예측")
    st.write("연구 결과를 확인하거나 X형 가새의 크기에 따른 구조응답을 예측할 수 있습니다.")
    st.divider()
    if st.button("① 가새 형태별 구조해석 결과"):
        go("shape")
    if st.button("② X형 가새 크기에 따른 AI 예측"):
        go("predict")
    if st.button("③ 논문 원본"):
        go("paper")
    if st.button("④ 모델 사진"):
        go("photos")
    st.caption("본 서비스는 연구용으로, 실제 구조설계의 적합성 판단을 대신하지 않습니다.")

elif st.session_state.page == "shape":
    st.header("가새 형태별 구조해석 결과")
    st.info("X형·K형·역V형의 실제 MIDAS 해석 결과를 비교하는 화면입니다. AI 예측을 사용하지 않습니다.")
    st.write("형태별 실제 해석값이 제공되면 이 화면에 원본 데이터를 표시하겠습니다.")
    back()

elif st.session_state.page == "predict":
    st.header("X형 가새 크기에 따른 AI 예측")
    st.caption("가새 형태: X형 고정 | 입력: 높이 A, 폭 B (m)")
    st.warning("학습된 11개 크기 조건을 바탕으로 한 연구용 추정값입니다. 학습 범위 내 미학습 조합도 검증된 값은 아닙니다.")
    with st.form("prediction"):
        height = st.number_input("가새 높이 A (m)", min_value=4.0, max_value=7.0, value=5.5, step=0.25, format="%.2f")
        width = st.number_input("가새 폭 B (m)", min_value=4.0, max_value=6.25, value=4.25, step=0.25, format="%.2f")
        submitted = st.form_submit_button("구조응답 예측하기", use_container_width=True)
    if submitted:
        try:
            models = load_models()
            inputs = pd.DataFrame({"가새형태": [0], "size_h": [height], "size_b": [width]})
            results = {name: float(model.predict(inputs)[0]) for name, model in models.items()}
            st.subheader("AI 예측 결과")
            st.metric("고유주기", f"{results['period']:.4f} sec")
            a, b = st.columns(2)
            a.metric("최대밑면전단력", f"{results['shear']:,.2f} kN")
            b.metric("최대층간변위비", f"{results['drift']:.6f}")
            st.metric("최대변위", f"{results['disp'] * 1000:.3f} mm")
            st.caption("최대변위는 학습 모델 출력 단위(m)를 mm로 환산했습니다.")
        except FileNotFoundError:
            st.error("모델 파일 4개가 저장소 최상위 폴더에 있는지 확인해주세요.")
        except Exception as e:
            st.error(f"모델 실행 오류: {e}")
    back()

elif st.session_state.page == "paper":
    st.header("논문 원본")
    pdf_path = BASE / "paper.pdf.pdf"
    if pdf_path.exists():
        import base64
       st.pdf(pdf_path.read_bytes(), height=720)
        st.download_button("PDF 다운로드 / 별도 열람", data=pdf_path.read_bytes(), file_name="paper.pdf", mime="application/pdf")
        st.caption("일부 모바일 브라우저에서는 내장 PDF 스크롤이 지원되지 않을 수 있습니다.")
    else:
        st.info("논문 원본 PDF가 준비되면 이곳에서 스크롤하여 열람할 수 있습니다.")
    back()

elif st.session_state.page == "photos":
    st.header("구조해석 모델 사진")

    for filename in ["model_1.png", "model_2.png", "model_3.png"]:
        image_path = BASE / filename

        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.warning(f"{filename} 파일을 찾을 수 없습니다.")

    back()
