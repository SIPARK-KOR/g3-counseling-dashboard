import ast
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


DATA_FILE = Path("counseling_records.csv")

STATUS_OPTIONS = ["학생부종합", "학생부교과", "논술", "정시", "면접 중심", "실기/특기", "기타"]
LEVEL_OPTIONS = ["상", "중상", "중", "중하", "하"]
GRADE_OPTIONS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
SEMESTERS = ["1-1", "1-2", "2-1", "2-2", "3-1", "3-2"]
SUBJECTS = ["국어", "수학", "영어", "사회", "과학", "교과평균"]

BASE_COLUMNS = [
    "timestamp", "student_code", "student_status", "desired_university_line",
    "desired_major_1", "desired_major_2", "priority_type", "career_decision",
    "gpa", "korean_gpa", "math_gpa", "english_gpa", "social_gpa", "science_gpa",
    "mock_korean", "mock_math", "mock_english", "mock_inquiry", "score_trend",
    "subject_record", "club_activity", "career_activity", "reading_activity",
    "academic_level", "career_level", "community_level",
    "upper_universities", "target_universities", "safe_universities", "admission_strategy",
    "strengths", "improvements", "memo", "next_plan"
]
SEMESTER_COLUMNS = [f"{subject}_{semester}" for subject in SUBJECTS for semester in SEMESTERS]
COLUMNS = BASE_COLUMNS + SEMESTER_COLUMNS


# ------------------------------
# 데이터 처리 함수
# ------------------------------
def load_data() -> pd.DataFrame:
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)


def save_record(record: dict) -> None:
    df = load_data()
    new_df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    new_df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")


def delete_records_by_ids(record_ids: list[int]) -> None:
    df = load_data()
    df = df.drop(index=record_ids)
    df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")


def get_latest_record(student_code: str) -> dict | None:
    df = load_data()
    if df.empty or not student_code.strip():
        return None

    matched = df[df["student_code"].astype(str) == student_code.strip()].copy()
    if matched.empty:
        return None

    if "timestamp" in matched.columns:
        matched = matched.sort_values(by="timestamp", ascending=False)

    return matched.iloc[0].to_dict()


def parse_status(value) -> list[str]:
    if isinstance(value, list):
        return [v for v in value if v in STATUS_OPTIONS]

    if pd.isna(value) or str(value).strip() == "":
        return ["학생부종합"]

    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [v for v in parsed if v in STATUS_OPTIONS]
    except Exception:
        pass

    return [option for option in STATUS_OPTIONS if option in text] or ["학생부종합"]


def get_loaded_value(record: dict | None, key: str, default):
    if not record:
        return default
    value = record.get(key, default)
    if pd.isna(value) or value == "":
        return default
    return value


def get_float_value(record: dict | None, key: str, default: float) -> float:
    try:
        return float(get_loaded_value(record, key, default))
    except Exception:
        return default


def get_text_value(record: dict | None, key: str, default: str = "") -> str:
    value = get_loaded_value(record, key, default)
    return str(value) if value is not None else default


def text_to_float_or_none(value: str):
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


# ------------------------------
# 요약 및 체크리스트 함수
# ------------------------------
def generate_summary(record: dict) -> str:
    lines = []
    status_text = ", ".join(record["student_status"]) if record["student_status"] else "미입력"

    lines.append(
        f"{record['student_code']} 학생은 {record['desired_major_1']} 계열을 1지망으로 희망하며, "
        f"희망 대학 라인은 {record['desired_university_line']}이다."
    )
    lines.append(
        f"지원 성향은 {status_text}이며, 진로 결정 상태는 {record['career_decision']}이다. "
        f"상담 우선순위는 {record['priority_type']}로 볼 수 있다."
    )
    lines.append(
        f"현재 내신 평균은 {record['gpa']}, 국어 {record['korean_gpa']}, 수학 {record['math_gpa']}, "
        f"영어 {record['english_gpa']}, 사회 {record['social_gpa']}, 과학 {record['science_gpa']} 수준이다."
    )
    lines.append(
        f"모의고사 등급은 국어 {record['mock_korean']}, 수학 {record['mock_math']}, "
        f"영어 {record['mock_english']}, 탐구 {record['mock_inquiry']} 수준이며, 최근 성적 흐름은 {record['score_trend']}이다."
    )
    lines.append(
        f"역량 평가는 학업역량 {record['academic_level']}, 진로역량 {record['career_level']}, "
        f"공동체역량 {record['community_level']}로 정리된다."
    )

    if str(record["strengths"]).strip():
        lines.append(f"강점은 {str(record['strengths']).strip()}이다.")
    if str(record["improvements"]).strip():
        lines.append(f"보완점은 {str(record['improvements']).strip()}이다.")
    if str(record["admission_strategy"]).strip():
        lines.append(f"지원 전략은 {str(record['admission_strategy']).strip()}를 중심으로 검토한다.")

    return "\n".join(lines)


def generate_checklist(record: dict) -> list[str]:
    text = " ".join([
        str(record["improvements"]), str(record["memo"]), str(record["subject_record"]),
        str(record["club_activity"]), str(record["career_activity"]), str(record["next_plan"])
    ])
    checklist = []

    if record["priority_type"] == "대학 우선":
        checklist.append("희망 대학 라인에 맞는 지원 가능 전형과 학과 허용 범위 확인")
    elif record["priority_type"] == "전공 우선":
        checklist.append(f"1지망 전공({record['desired_major_1']})과 학생부 활동의 연결성 확인")
    else:
        checklist.append("대학 라인과 희망 전공 사이의 우선순위 재확인")

    if record["career_decision"] == "탐색 중":
        checklist.append("진로 방향을 좁히기 위한 전공 비교 상담 필요")
    if record["academic_level"] in ["중하", "하"] or record["score_trend"] == "하락":
        checklist.append("최근 성적 변화 원인과 남은 기간 학습 전략 점검")
    if str(record["subject_record"]).strip() or "세특" in text:
        checklist.append("교과 세특에서 전공 관련 탐구가 구체적으로 드러나는지 확인")
    if str(record["club_activity"]).strip() or "동아리" in text:
        checklist.append("동아리 활동에서 학생의 역할과 성장 과정 정리")
    if "면접" in text or "학종" in record["student_status"]:
        checklist.append("면접에서 말할 핵심 활동 2~3개 선정")
    if str(record["upper_universities"]).strip() or str(record["target_universities"]).strip() or str(record["safe_universities"]).strip():
        checklist.append("상향·적정·안정 지원군의 균형 재검토")
    if str(record["next_plan"]).strip():
        checklist.append(f"다음 상담 전 준비 과제 확인: {str(record['next_plan']).strip()}")

    return checklist


# ------------------------------
# 시각화 함수
# ------------------------------
def color_level(level: str) -> str:
    color_map = {
        "상": "🟢 상",
        "중상": "🟡 중상",
        "중": "🟠 중",
        "중하": "🔴 중하",
        "하": "⚫ 하"
    }
    return color_map.get(level, level)


def plot_semester_grade_charts(record: dict) -> None:
    st.markdown("**학기별 성적 변화 시각화**")

    chart_data = []
    for subject in SUBJECTS:
        for semester in SEMESTERS:
            key = f"{subject}_{semester}"
            value = text_to_float_or_none(record.get(key, ""))
            if value is not None:
                chart_data.append({"과목": subject, "학기": semester, "등급": value})

    if not chart_data:
        st.info("학기별 성적 데이터가 입력되면 그래프가 표시된다.")
        return

    df = pd.DataFrame(chart_data)
    col_a, col_b, col_c = st.columns(3)
    columns = [col_a, col_b, col_c]

    for idx, subject in enumerate(SUBJECTS):
        subject_df = df[df["과목"] == subject]
        if subject_df.empty:
            continue

        with columns[idx % 3]:
            fig = px.line(subject_df, x="학기", y="등급", markers=True, title=subject)
            fig.update_yaxes(range=[9, 1], title="등급")
            fig.update_xaxes(categoryorder="array", categoryarray=SEMESTERS)
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig, use_container_width=True)


# ------------------------------
# 구조화된 AI 분석 결과 자동 입력 함수
# ------------------------------
def parse_pipe_format(text: str) -> dict:
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        key, value = line.split("|", 1)
        result[key.strip()] = value.strip()
    return result


def clean_grade(value: str, default: float = 2.50) -> float:
    text = str(value)
    number = ""
    dot_used = False

    for char in text:
        if char.isdigit():
            number += char
        elif char == "." and not dot_used and number:
            number += char
            dot_used = True
        elif number:
            break

    if number:
        return float(number)
    return default


def clean_level(value: str, default: str = "중") -> str:
    text = str(value)
    for level in ["중상", "중하", "상", "중", "하"]:
        if level in text:
            return level
    return default


def clean_flow(value: str) -> str:
    text = str(value)
    if "상승" in text:
        return "상승"
    if "하락" in text:
        return "하락"
    if "유지" in text:
        return "유지"
    return "판단 보류"


def parse_major(value: str) -> tuple[str, str]:
    text = str(value)
    for sep in [",", "/", "·", "|"]:
        text = text.replace(sep, ",")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    major_1 = parts[0] if parts else text
    major_2 = parts[1] if len(parts) > 1 else ""
    return major_1, major_2


def parse_ai_analysis(text: str) -> dict:
    data = parse_pipe_format(text)

    upper = data.get("상향대학", "")
    target = data.get("적정대학", "")
    safe = data.get("안정대학", "")
    university_line = data.get("희망대학라인", "") or f"상향-{upper} / 적정-{target} / 안정-{safe}"

    desired_major_1, desired_major_2 = parse_major(data.get("희망전공", ""))

    parsed = {
        "student_status": ["학생부종합"],
        "desired_university_line": university_line,
        "desired_major_1": desired_major_1,
        "desired_major_2": desired_major_2,
        "priority_type": "대학·전공 균형",
        "career_decision": "어느 정도 확정",
        "upper_universities": upper,
        "target_universities": target,
        "safe_universities": safe,
        "admission_strategy": f"상향-{upper} / 적정-{target} / 안정-{safe}",
        "gpa": clean_grade(data.get("내신평균", "")),
        "korean_gpa": clean_grade(data.get("국어내신", "")),
        "math_gpa": clean_grade(data.get("수학내신", "")),
        "english_gpa": clean_grade(data.get("영어내신", "")),
        "social_gpa": clean_grade(data.get("사회내신", "")),
        "science_gpa": clean_grade(data.get("과학내신", "")),
        "mock_korean": "1",
        "mock_math": "1",
        "mock_english": "1",
        "mock_inquiry": "1",
        "score_trend": clean_flow(data.get("성적흐름", "")),
        "subject_record": data.get("교과세특", "정보 부족"),
        "club_activity": data.get("동아리", "정보 부족"),
        "career_activity": data.get("진로활동", "정보 부족"),
        "reading_activity": data.get("독서기타", "정보 부족"),
        "academic_level": clean_level(data.get("학업역량", "")),
        "career_level": clean_level(data.get("진로역량", "")),
        "community_level": clean_level(data.get("공동체역량", "")),
        "strengths": data.get("강점", "정보 부족"),
        "improvements": data.get("보완점", "정보 부족"),
        "memo": data.get("상담메모", ""),
        "next_plan": data.get("다음준비", "정보 부족")
    }

    for subject in SUBJECTS:
        for semester in SEMESTERS:
            key = f"{subject}_{semester}"
            parsed[key] = data.get(key, "")

    return parsed


# ------------------------------
# 로그인 함수
# ------------------------------
def check_password() -> bool:
    correct_password = "1234"

    if "password_ok" not in st.session_state:
        st.session_state["password_ok"] = False

    if st.session_state["password_ok"]:
        return True

    st.title("진학 상담 기록 및 분석 시스템")
    st.write("이 시스템은 상담 기록 보호를 위해 비밀번호 입력 후 사용할 수 있다.")

    password = st.text_input("비밀번호", type="password")

    if st.button("로그인"):
        if password == correct_password:
            st.session_state["password_ok"] = True
            st.success("로그인되었습니다.")
            st.rerun()
        else:
            st.error("비밀번호가 올바르지 않습니다.")

    return False


# ------------------------------
# 화면 구성
# ------------------------------
def main() -> None:
    st.set_page_config(page_title="진학 상담 기록 대시보드", layout="wide")

    if not check_password():
        st.stop()

    st.title("진학 상담 기록 및 분석 대시보드")
    st.write("담임 및 진학 담당 교사가 학생 상담 기록을 저장하고 요약할 수 있는 상담 지원 시스템이다.")

    with st.sidebar:
        st.header("안내")
        if st.button("로그아웃"):
            st.session_state["password_ok"] = False
            st.rerun()
        st.write("- 기존 학생 상담 기록을 불러온 뒤 일부만 수정하여 새 상담 기록으로 저장할 수 있다.")
        st.write("- 학생 개인정보 보호를 위해 학생 이름 대신 학생 코드를 사용한다.")
        st.write("- AI 분석 결과를 구조화된 형식으로 붙여넣으면 입력칸에 자동 반영된다.")

    st.subheader("0. 기존 상담 기록 불러오기")
    load_col1, load_col2 = st.columns([2, 1])
    with load_col1:
        load_code = st.text_input("불러올 학생 코드", placeholder="예: 3101")
    with load_col2:
        st.write("")
        st.write("")
        if st.button("최신 상담 기록 불러오기", use_container_width=True):
            loaded = get_latest_record(load_code)
            if loaded is None:
                st.warning("해당 학생 코드의 상담 기록이 없습니다.")
                st.session_state["loaded_record"] = None
            else:
                st.session_state["loaded_record"] = loaded
                st.success("최신 상담 기록을 불러왔습니다. 아래 입력칸에서 필요한 부분만 수정하세요.")
                st.rerun()

    loaded_record = st.session_state.get("loaded_record")
    if loaded_record:
        st.info(f"현재 {loaded_record.get('student_code', '')} 학생의 최신 상담 기록을 불러온 상태입니다. 수정 후 저장하면 새 상담 기록으로 추가됩니다.")
        if st.button("불러온 내용 초기화"):
            st.session_state["loaded_record"] = None
            st.rerun()

    st.subheader("0-1. AI 분석 결과 자동 분류")
    ai_text = st.text_area(
        "AI가 정리한 학생부 분석 결과 붙여넣기",
        placeholder="항목명|내용 형식의 AI 분석 결과를 여기에 붙여넣고 버튼을 누르면 입력칸에 자동 반영됩니다.",
        height=180
    )

    if st.button("AI 분석 결과를 입력칸에 반영", use_container_width=True):
        if not ai_text.strip():
            st.warning("AI 분석 결과 텍스트를 먼저 붙여넣어 주세요.")
        else:
            parsed_record = parse_ai_analysis(ai_text)

            st.session_state["loaded_record"] = parsed_record

            for subject in SUBJECTS:
                for semester in SEMESTERS:
                    key = f"{subject}_{semester}"
                    widget_key = f"input_{key}"
                    st.session_state[widget_key] = parsed_record.get(key, "")

            st.success("AI 분석 결과를 입력칸에 반영했습니다. 아래 항목을 확인하고 필요한 부분을 수정하세요.")
            st.rerun()

    st.subheader("1. 학생 기본 정보")
    col1, col2, col3 = st.columns(3)
    with col1:
        student_code = st.text_input("학생 코드", value=get_text_value(loaded_record, "student_code", ""), placeholder="예: 3122")
        career_options = ["확정", "어느 정도 확정", "탐색 중"]
        career_default = get_text_value(loaded_record, "career_decision", "확정")
        career_decision = st.selectbox("진로 결정 상태", career_options, index=career_options.index(career_default) if career_default in career_options else 0)
    with col2:
        student_status = st.multiselect("학생 준비 유형", STATUS_OPTIONS, default=parse_status(get_loaded_value(loaded_record, "student_status", ["학생부종합"])))
        priority_options = ["전공 우선", "대학 우선", "대학·전공 균형", "아직 모름"]
        priority_default = get_text_value(loaded_record, "priority_type", "전공 우선")
        priority_type = st.selectbox("상담 우선순위", priority_options, index=priority_options.index(priority_default) if priority_default in priority_options else 0)
    with col3:
        desired_university_line = st.text_input("희망 대학 라인", value=get_text_value(loaded_record, "desired_university_line", ""), placeholder="예: 부산권 국립대, 수도권 중상위권")
        desired_major_1 = st.text_input("희망 전공 1지망", value=get_text_value(loaded_record, "desired_major_1", ""), placeholder="예: 컴퓨터공학")
        desired_major_2 = st.text_input("희망 전공 2지망", value=get_text_value(loaded_record, "desired_major_2", ""), placeholder="예: 산업공학")

    st.subheader("2. 성적 정보")
    col4, col5, col6 = st.columns(3)
    with col4:
        gpa = st.number_input("내신 평균 등급", min_value=1.0, max_value=9.0, value=get_float_value(loaded_record, "gpa", 2.50), step=0.01, format="%.2f")
        korean_gpa = st.number_input("국어 내신 등급", min_value=1.0, max_value=9.0, value=get_float_value(loaded_record, "korean_gpa", 2.50), step=0.01, format="%.2f")
    with col5:
        math_gpa = st.number_input("수학 내신 등급", min_value=1.0, max_value=9.0, value=get_float_value(loaded_record, "math_gpa", 2.50), step=0.01, format="%.2f")
        english_gpa = st.number_input("영어 내신 등급", min_value=1.0, max_value=9.0, value=get_float_value(loaded_record, "english_gpa", 2.50), step=0.01, format="%.2f")
    with col6:
        social_gpa = st.number_input("사회 내신 등급", min_value=1.0, max_value=9.0, value=get_float_value(loaded_record, "social_gpa", 2.50), step=0.01, format="%.2f")
        science_gpa = st.number_input("과학 내신 등급", min_value=1.0, max_value=9.0, value=get_float_value(loaded_record, "science_gpa", 2.50), step=0.01, format="%.2f")

    col7, col8 = st.columns(2)
    with col7:
        trend_options = ["상승", "유지", "하락", "판단 보류"]
        trend_default = get_text_value(loaded_record, "score_trend", "유지")
        score_trend = st.selectbox("최근 성적 흐름", trend_options, index=trend_options.index(trend_default) if trend_default in trend_options else 1)
    with col8:
        mock_korean_default = get_text_value(loaded_record, "mock_korean", "1")
        mock_math_default = get_text_value(loaded_record, "mock_math", "1")
        mock_english_default = get_text_value(loaded_record, "mock_english", "1")
        mock_inquiry_default = get_text_value(loaded_record, "mock_inquiry", "1")

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            mock_korean = st.selectbox(
                "모의고사 국어",
                GRADE_OPTIONS,
                index=GRADE_OPTIONS.index(mock_korean_default) if mock_korean_default in GRADE_OPTIONS else 0
            )

        with m2:
            mock_math = st.selectbox(
                "모의고사 수학",
                GRADE_OPTIONS,
                index=GRADE_OPTIONS.index(mock_math_default) if mock_math_default in GRADE_OPTIONS else 0
            )

        with m3:
            mock_english = st.selectbox(
                "모의고사 영어",
                GRADE_OPTIONS,
                index=GRADE_OPTIONS.index(mock_english_default) if mock_english_default in GRADE_OPTIONS else 0
            )

        with m4:
            mock_inquiry = st.selectbox(
                "모의고사 탐구",
                GRADE_OPTIONS,
                index=GRADE_OPTIONS.index(mock_inquiry_default) if mock_inquiry_default in GRADE_OPTIONS else 0
            )

    st.subheader("2-1. 학기별 성적 입력")
    st.caption("사회·과학은 해당 학기에 이수한 여러 과목의 평균 등급을 입력한다. 예: 물리학 2등급, 지구과학 3등급 → 과학 2.5등급")

    semester_values = {}
    with st.expander("학기별 성적 입력 열기"):
        for subject in SUBJECTS:
            st.markdown(f"**{subject}**")
            cols = st.columns(6)
            for idx, semester in enumerate(SEMESTERS):
                key = f"{subject}_{semester}"
                widget_key = f"input_{key}"

                if widget_key not in st.session_state:
                    st.session_state[widget_key] = get_text_value(loaded_record, key, "")

                with cols[idx]:
                    semester_values[key] = st.text_input(
                        semester,
                        key=widget_key,
                        placeholder="예: 2.5"
                    )

    st.subheader("3. 학생부 및 활동 핵심 내용")
    col9, col10 = st.columns(2)
    with col9:
        subject_record = st.text_area("교과 세특 핵심 내용", value=get_text_value(loaded_record, "subject_record", ""), height=100)
        club_activity = st.text_area("동아리 활동", value=get_text_value(loaded_record, "club_activity", ""), height=100)
    with col10:
        career_activity = st.text_area("진로 활동", value=get_text_value(loaded_record, "career_activity", ""), height=100)
        reading_activity = st.text_area("독서 및 기타 활동", value=get_text_value(loaded_record, "reading_activity", ""), height=100)

    st.subheader("4. 학생부종합전형 역량 판단")
    academic_default = get_text_value(loaded_record, "academic_level", "중")
    career_default = get_text_value(loaded_record, "career_level", "중")
    community_default = get_text_value(loaded_record, "community_level", "중")
    col11, col12, col13 = st.columns(3)
    with col11:
        academic_level = st.selectbox("학업역량", LEVEL_OPTIONS, index=LEVEL_OPTIONS.index(academic_default) if academic_default in LEVEL_OPTIONS else 2)
    with col12:
        career_level = st.selectbox("진로역량", LEVEL_OPTIONS, index=LEVEL_OPTIONS.index(career_default) if career_default in LEVEL_OPTIONS else 2)
    with col13:
        community_level = st.selectbox("공동체역량", LEVEL_OPTIONS, index=LEVEL_OPTIONS.index(community_default) if community_default in LEVEL_OPTIONS else 2)

    st.subheader("5. 지원 전략")
    col14, col15, col16 = st.columns(3)
    with col14:
        upper_universities = st.text_area("상향 지원 후보", value=get_text_value(loaded_record, "upper_universities", ""), height=90)
    with col15:
        target_universities = st.text_area("적정 지원 후보", value=get_text_value(loaded_record, "target_universities", ""), height=90)
    with col16:
        safe_universities = st.text_area("안정 지원 후보", value=get_text_value(loaded_record, "safe_universities", ""), height=90)

    admission_strategy = st.text_area("지원 전략 메모", value=get_text_value(loaded_record, "admission_strategy", ""), height=100)

    st.subheader("6. 상담 기록")
    col17, col18 = st.columns(2)
    with col17:
        strengths = st.text_area("학생 강점", value=get_text_value(loaded_record, "strengths", ""), height=100)
        improvements = st.text_area("보완점", value=get_text_value(loaded_record, "improvements", ""), height=100)
    with col18:
        next_plan = st.text_area("다음 상담 전 준비할 내용", value=get_text_value(loaded_record, "next_plan", ""), height=100)

    memo = st.text_area("상담 메모", value=get_text_value(loaded_record, "memo", ""), height=140)

    if st.button("상담 기록 저장", use_container_width=True):
        if not student_code.strip() or not desired_major_1.strip() or not memo.strip():
            st.warning("학생 코드, 희망 전공 1지망, 상담 메모는 반드시 입력해야 한다.")
        else:
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "student_code": student_code.strip(),
                "student_status": student_status,
                "desired_university_line": desired_university_line.strip(),
                "desired_major_1": desired_major_1.strip(),
                "desired_major_2": desired_major_2.strip(),
                "priority_type": priority_type,
                "career_decision": career_decision,
                "gpa": gpa,
                "korean_gpa": korean_gpa,
                "math_gpa": math_gpa,
                "english_gpa": english_gpa,
                "social_gpa": social_gpa,
                "science_gpa": science_gpa,
                "mock_korean": mock_korean,
                "mock_math": mock_math,
                "mock_english": mock_english,
                "mock_inquiry": mock_inquiry,
                "score_trend": score_trend,
                "subject_record": subject_record.strip(),
                "club_activity": club_activity.strip(),
                "career_activity": career_activity.strip(),
                "reading_activity": reading_activity.strip(),
                "academic_level": academic_level,
                "career_level": career_level,
                "community_level": community_level,
                "upper_universities": upper_universities.strip(),
                "target_universities": target_universities.strip(),
                "safe_universities": safe_universities.strip(),
                "admission_strategy": admission_strategy.strip(),
                "strengths": strengths.strip(),
                "improvements": improvements.strip(),
                "memo": memo.strip(),
                "next_plan": next_plan.strip(),
            }
            record.update(semester_values)
            save_record(record)
            st.success("상담 기록이 저장되었다. 같은 학생 코드로 저장하면 상담 이력이 누적된다.")

    st.subheader("7. 학생 핵심 요약 대시보드")
    card1, card2, card3, card4 = st.columns(4)
    with card1:
        st.metric("내신 평균", f"{gpa:.2f}")
    with card2:
        st.metric("수학 내신", f"{math_gpa:.2f}")
    with card3:
        st.metric("희망 대학 라인", desired_university_line if desired_university_line else "미입력")
    with card4:
        st.metric("희망 전공", desired_major_1 if desired_major_1 else "미입력")

    dash_col1, dash_col2 = st.columns([1, 2])
    with dash_col1:
        st.markdown("**역량 한눈에 보기**")
        st.markdown(f"- 학업역량: {color_level(academic_level)}")
        st.markdown(f"- 진로역량: {color_level(career_level)}")
        st.markdown(f"- 공동체역량: {color_level(community_level)}")
        st.markdown(f"- 최근 성적 흐름: **{score_trend}**")
    with dash_col2:
        current_chart_record = {}
        if loaded_record:
            current_chart_record.update(loaded_record)
        current_chart_record.update(semester_values)
        plot_semester_grade_charts(current_chart_record)

    st.subheader("8. 상담 요약 및 다음 상담 체크리스트")
    current_record = {
        "student_code": student_code.strip(),
        "student_status": student_status,
        "desired_university_line": desired_university_line.strip(),
        "desired_major_1": desired_major_1.strip(),
        "priority_type": priority_type,
        "career_decision": career_decision,
        "gpa": gpa,
        "korean_gpa": korean_gpa,
        "math_gpa": math_gpa,
        "english_gpa": english_gpa,
        "social_gpa": social_gpa,
        "science_gpa": science_gpa,
        "mock_korean": mock_korean,
        "mock_math": mock_math,
        "mock_english": mock_english,
        "mock_inquiry": mock_inquiry,
        "score_trend": score_trend,
        "subject_record": subject_record.strip(),
        "club_activity": club_activity.strip(),
        "career_activity": career_activity.strip(),
        "academic_level": academic_level,
        "career_level": career_level,
        "community_level": community_level,
        "upper_universities": upper_universities.strip(),
        "target_universities": target_universities.strip(),
        "safe_universities": safe_universities.strip(),
        "admission_strategy": admission_strategy.strip(),
        "strengths": strengths.strip(),
        "improvements": improvements.strip(),
        "memo": memo.strip(),
        "next_plan": next_plan.strip(),
    }

    summary = generate_summary(current_record)
    checklist = generate_checklist(current_record)

    col19, col20 = st.columns(2)
    with col19:
        st.markdown("**상담 요약**")
        st.text(summary if memo.strip() else "입력 후 요약이 표시된다.")
    with col20:
        st.markdown("**다음 상담 체크리스트**")
        if memo.strip():
            for item in checklist:
                st.write(f"- {item}")
        else:
            st.write("입력 후 체크리스트가 표시된다.")

    st.subheader("9. 전체 학생 비교 분석")
    df = load_data()

    if df.empty:
        st.info("아직 저장된 상담 기록이 없습니다.")
    else:
        analysis_df = df.copy()
        numeric_columns = ["gpa", "korean_gpa", "math_gpa", "english_gpa", "social_gpa", "science_gpa"]
        for col in numeric_columns:
            if col in analysis_df.columns:
                analysis_df[col] = pd.to_numeric(analysis_df[col], errors="coerce")

        st.markdown("**필터 조건**")
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)

        with f_col1:
            gpa_range = st.slider("내신 평균 범위", 1.0, 9.0, (1.0, 9.0), step=0.1)
        with f_col2:
            major_keyword = st.text_input("희망 전공 검색", placeholder="예: 컴퓨터, 간호, 경영")
        with f_col3:
            university_keyword = st.text_input("희망 대학 라인 검색", placeholder="예: 인서울, 부산권, 국립대")
        with f_col4:
            status_filter = st.multiselect("준비 유형 필터", STATUS_OPTIONS)

        filtered_analysis = analysis_df.copy()

        if "gpa" in filtered_analysis.columns:
            filtered_analysis = filtered_analysis[(filtered_analysis["gpa"] >= gpa_range[0]) & (filtered_analysis["gpa"] <= gpa_range[1])]

        if major_keyword.strip() and "desired_major_1" in filtered_analysis.columns:
            filtered_analysis = filtered_analysis[
                filtered_analysis["desired_major_1"].astype(str).str.contains(major_keyword.strip(), case=False, na=False) |
                filtered_analysis["desired_major_2"].astype(str).str.contains(major_keyword.strip(), case=False, na=False)
            ]

        if university_keyword.strip() and "desired_university_line" in filtered_analysis.columns:
            filtered_analysis = filtered_analysis[filtered_analysis["desired_university_line"].astype(str).str.contains(university_keyword.strip(), case=False, na=False)]

        if status_filter and "student_status" in filtered_analysis.columns:
            status_pattern = "|".join(status_filter)
            filtered_analysis = filtered_analysis[filtered_analysis["student_status"].astype(str).str.contains(status_pattern, na=False)]

        st.markdown("**필터 결과 요약**")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("해당 학생 수", filtered_analysis["student_code"].nunique())

        view_columns = [
            "timestamp", "student_code", "student_status", "desired_university_line",
            "desired_major_1", "desired_major_2", "priority_type", "gpa", "math_gpa",
            "social_gpa", "science_gpa", "academic_level", "career_level", "community_level",
            "upper_universities", "target_universities", "safe_universities", "next_plan"
        ]
        existing_view_columns = [col for col in view_columns if col in filtered_analysis.columns]
        st.dataframe(filtered_analysis[existing_view_columns], use_container_width=True)

        latest_by_student = filtered_analysis.sort_values("timestamp").drop_duplicates("student_code", keep="last")

        st.markdown("**그룹별 간단 비교**")
        group_col1, group_col2 = st.columns(2)

        with group_col1:
            if "desired_major_1" in latest_by_student.columns and not latest_by_student.empty:
                major_counts = latest_by_student["desired_major_1"].value_counts().reset_index()
                major_counts.columns = ["희망 전공 1지망", "학생 수"]
                st.dataframe(major_counts, use_container_width=True)

        with group_col2:
            if "desired_university_line" in latest_by_student.columns and not latest_by_student.empty:
                line_counts = latest_by_student["desired_university_line"].value_counts().reset_index()
                line_counts.columns = ["희망 대학 라인", "학생 수"]
                st.dataframe(line_counts, use_container_width=True)

    st.subheader("10. 저장된 상담 기록 조회 및 삭제")
    search_code = st.text_input("조회할 학생 코드", placeholder="예: 3122")

    if search_code.strip():
        filtered = df[df["student_code"].astype(str) == search_code.strip()].copy()
    else:
        filtered = df.copy()

    if "timestamp" in filtered.columns and not filtered.empty:
        filtered = filtered.sort_values(by="timestamp", ascending=False)

    if filtered.empty:
        st.info("저장된 상담 기록이 없습니다.")
    else:
        display_df = filtered.copy()
        display_df.insert(0, "삭제 선택", False)

        edited_df = st.data_editor(
            display_df,
            use_container_width=True,
            hide_index=False,
            disabled=[col for col in display_df.columns if col != "삭제 선택"],
            key="delete_table"
        )

        selected_indexes = edited_df[edited_df["삭제 선택"]].index.tolist()

        if selected_indexes:
            st.warning(f"선택한 상담 기록 {len(selected_indexes)}건을 삭제할 수 있습니다.")

        if st.button("선택한 상담 기록 삭제", type="primary"):
            if selected_indexes:
                delete_records_by_ids(selected_indexes)
                st.success("선택한 상담 기록이 삭제되었습니다.")
                st.rerun()
            else:
                st.warning("삭제할 기록을 먼저 선택하세요.")


if __name__ == "__main__":
    main()
