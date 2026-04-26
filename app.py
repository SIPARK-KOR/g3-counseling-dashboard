import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


DATA_FILE = Path("counseling_records.csv")
STOPWORDS = {
    "학생", "상담", "활동", "계획", "필요", "부분", "내용", "결과", "학과", "전공",
    "대한", "통해", "위한", "관련", "이번", "다음", "현재", "조금", "매우", "있음"
}


# ------------------------------
# 데이터 처리 함수
# ------------------------------
def load_data() -> pd.DataFrame:
    """저장된 상담 데이터를 불러온다."""
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)

    columns = [
        "timestamp", "student_code", "student_status", "desired_university_line",
        "desired_major_1", "desired_major_2", "priority_type", "career_decision",
        "gpa", "korean_gpa", "math_gpa", "english_gpa", "inquiry_gpa",
        "mock_korean", "mock_math", "mock_english", "mock_inquiry", "score_trend",
        "subject_record", "club_activity", "career_activity", "reading_activity",
        "academic_level", "career_level", "community_level",
        "upper_universities", "target_universities", "safe_universities", "admission_strategy",
        "strengths", "improvements", "memo", "next_plan"
    ]
    return pd.DataFrame(columns=columns)


def save_record(record: dict) -> None:
    """상담 기록 1건을 CSV 파일에 저장한다."""
    df = load_data()
    new_df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    new_df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")


def delete_records_by_ids(record_ids: list[int]) -> None:
    """선택한 행 번호의 상담 기록을 삭제한다."""
    df = load_data()
    df = df.drop(index=record_ids)
    df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")


# ------------------------------
# 분석 함수
# ------------------------------
def extract_keywords(text: str) -> list[str]:
    """상담 메모에서 2글자 이상 한글 단어를 추출한다."""
    words = re.findall(r"[가-힣]{2,}", text)
    return [word for word in words if word not in STOPWORDS]


def get_keyword_counts(text: str, top_n: int = 10) -> pd.DataFrame:
    """핵심 단어 빈도를 데이터프레임으로 반환한다."""
    keywords = extract_keywords(text)
    counts = Counter(keywords)
    return pd.DataFrame(counts.most_common(top_n), columns=["키워드", "빈도"])


# ------------------------------
# 요약 생성 함수
# ------------------------------
def generate_summary(record: dict) -> str:
    """입력 내용을 바탕으로 진학 상담 요약문을 생성한다."""
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
        f"영어 {record['english_gpa']}, 탐구 {record['inquiry_gpa']} 수준이다."
    )
    lines.append(
        f"모의고사 등급은 국어 {record['mock_korean']}, 수학 {record['mock_math']}, "
        f"영어 {record['mock_english']}, 탐구 {record['mock_inquiry']} 수준이며, 최근 성적 흐름은 {record['score_trend']}이다."
    )
    lines.append(
        f"역량 평가는 학업역량 {record['academic_level']}, 진로역량 {record['career_level']}, "
        f"공동체역량 {record['community_level']}로 정리된다."
    )

    if record["strengths"].strip():
        lines.append(f"강점은 {record['strengths'].strip()}이다.")

    if record["improvements"].strip():
        lines.append(f"보완점은 {record['improvements'].strip()}이다.")

    if record["admission_strategy"].strip():
        lines.append(f"지원 전략은 {record['admission_strategy'].strip()}를 중심으로 검토한다.")

    return "\n".join(lines)


def generate_checklist(record: dict) -> list[str]:
    """다음 상담에서 확인할 체크리스트를 생성한다."""
    text = " ".join([
        record["improvements"], record["memo"], record["subject_record"],
        record["club_activity"], record["career_activity"], record["next_plan"]
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

    if record["subject_record"].strip() or "세특" in text:
        checklist.append("교과 세특에서 전공 관련 탐구가 구체적으로 드러나는지 확인")

    if record["club_activity"].strip() or "동아리" in text:
        checklist.append("동아리 활동에서 학생의 역할과 성장 과정 정리")

    if "면접" in text or "학종" in record["student_status"]:
        checklist.append("면접에서 말할 핵심 활동 2~3개 선정")

    if record["upper_universities"].strip() or record["target_universities"].strip() or record["safe_universities"].strip():
        checklist.append("상향·적정·안정 지원군의 균형 재검토")

    if record["next_plan"].strip():
        checklist.append(f"다음 상담 전 준비 과제 확인: {record['next_plan'].strip()}")

    return checklist


# ------------------------------
# 시각화 함수
# ------------------------------
def plot_keyword_chart(keyword_df: pd.DataFrame) -> None:
    """키워드 빈도 막대그래프를 그린다."""
    if keyword_df.empty:
        st.info("표시할 키워드가 없습니다.")
        return

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(keyword_df["키워드"], keyword_df["빈도"])
    ax.set_title("상담 메모 핵심 키워드")
    ax.set_xlabel("키워드")
    ax.set_ylabel("빈도")
    plt.xticks(rotation=30)
    st.pyplot(fig)


# ------------------------------
# 화면 구성
# ------------------------------
def main() -> None:
    """Streamlit 메인 화면을 구성한다."""
    st.set_page_config(page_title="고3 진학 상담 기록 대시보드", layout="wide")
    st.title("고3 진학 상담 기록 및 키워드 분석 대시보드")
    st.write("고3 담임 및 진학 담당 교사가 학생 상담 기록을 저장하고 요약할 수 있는 1차 버전이다.")

    with st.sidebar:
        st.header("안내")
        st.write("- 1차 버전: 입력, 저장, 조회, 키워드 분석")
        st.write("- 학생 개인정보 보호를 위해 학생 이름 대신 학생 코드를 사용한다.")
        st.write("- 이후 버전: 로그인, DB 연동, AI 문장 생성 추가 가능")

    st.subheader("1. 학생 기본 정보")
    col1, col2, col3 = st.columns(3)
    with col1:
        student_code = st.text_input("학생 코드", placeholder="예: 3122")
        career_decision = st.selectbox("진로 결정 상태", ["확정", "어느 정도 확정", "탐색 중"])
    with col2:
        student_status = st.multiselect(
            "학생 준비 유형",
            ["학생부종합", "학생부교과", "논술", "정시", "면접 중심", "실기/특기", "기타"],
            default=["학생부종합"]
        )
        priority_type = st.selectbox("상담 우선순위", ["전공 우선", "대학 우선", "대학·전공 균형", "아직 모름"])
    with col3:
        desired_university_line = st.text_input("희망 대학 라인", placeholder="예: 인서울, 부산권 국립대, 수도권 중상위권")
        desired_major_1 = st.text_input("희망 전공 1지망", placeholder="예: 컴퓨터공학")
        desired_major_2 = st.text_input("희망 전공 2지망", placeholder="예: 산업공학")

    st.subheader("2. 성적 정보")
    col4, col5, col6, col7 = st.columns(4)
    with col4:
        gpa = st.number_input("내신 평균 등급", min_value=1.0, max_value=9.0, value=2.50, step=0.01, format="%.2f")
        korean_gpa = st.number_input("국어 내신 등급", min_value=1.0, max_value=9.0, value=2.50, step=0.01, format="%.2f")
    with col5:
        math_gpa = st.number_input("수학 내신 등급", min_value=1.0, max_value=9.0, value=2.50, step=0.01, format="%.2f")
        english_gpa = st.number_input("영어 내신 등급", min_value=1.0, max_value=9.0, value=2.50, step=0.01, format="%.2f")
    with col6:
        inquiry_gpa = st.number_input("탐구 내신 등급", min_value=1.0, max_value=9.0, value=2.50, step=0.01, format="%.2f")
        score_trend = st.selectbox("최근 성적 흐름", ["상승", "유지", "하락", "판단 보류"])
    with col7:
        mock_korean = st.selectbox("모의고사 국어", ["1", "2", "3", "4", "5", "6", "7", "8", "9"])
        mock_math = st.selectbox("모의고사 수학", ["1", "2", "3", "4", "5", "6", "7", "8", "9"])
        mock_english = st.selectbox("모의고사 영어", ["1", "2", "3", "4", "5", "6", "7", "8", "9"])
        mock_inquiry = st.selectbox("모의고사 탐구", ["1", "2", "3", "4", "5", "6", "7", "8", "9"])

    st.subheader("3. 학생부 및 활동 핵심 내용")
    col8, col9 = st.columns(2)
    with col8:
        subject_record = st.text_area("교과 세특 핵심 내용", placeholder="예: 수학 탐구, 통계 분석, 문제 해결 과정 등", height=100)
        club_activity = st.text_area("동아리 활동", placeholder="예: 역할, 활동 주제, 지속성", height=100)
    with col9:
        career_activity = st.text_area("진로 활동", placeholder="예: 전공 탐색, 진로 발표, 탐구 보고서", height=100)
        reading_activity = st.text_area("독서 및 기타 활동", placeholder="예: 전공 관련 독서, 봉사, 자율활동", height=100)

    st.subheader("4. 학생부종합전형 역량 판단")
    level_options = ["상", "중상", "중", "중하", "하"]
    col10, col11, col12 = st.columns(3)
    with col10:
        academic_level = st.selectbox("학업역량", level_options)
    with col11:
        career_level = st.selectbox("진로역량", level_options)
    with col12:
        community_level = st.selectbox("공동체역량", level_options)

    st.subheader("5. 지원 전략")
    col13, col14, col15 = st.columns(3)
    with col13:
        upper_universities = st.text_area("상향 지원 후보", placeholder="예: A대 ○○학과", height=90)
    with col14:
        target_universities = st.text_area("적정 지원 후보", placeholder="예: B대 ○○학과", height=90)
    with col15:
        safe_universities = st.text_area("안정 지원 후보", placeholder="예: C대 ○○학과", height=90)

    admission_strategy = st.text_area("지원 전략 메모", placeholder="예: 학생부종합 3장, 교과 2장, 논술 1장 등", height=100)

    st.subheader("6. 상담 기록")
    col16, col17 = st.columns(2)
    with col16:
        strengths = st.text_area("학생 강점", placeholder="예: 수학 탐구 역량, 꾸준한 학습 태도", height=100)
        improvements = st.text_area("보완점", placeholder="예: 전공 관련 활동의 구체성 부족", height=100)
    with col17:
        next_plan = st.text_area("다음 상담 전 준비할 내용", placeholder="예: 지원 희망 대학 5곳 조사, 세특 관련 탐구 주제 정리", height=100)

    memo = st.text_area("상담 메모", placeholder="이번 상담에서 나눈 핵심 내용을 입력하세요.", height=140)

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
                "inquiry_gpa": inquiry_gpa,
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
            save_record(record)
            st.success("상담 기록이 저장되었다.")

    st.subheader("7. 상담 요약 및 다음 상담 체크리스트")
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
        "inquiry_gpa": inquiry_gpa,
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

    col18, col19 = st.columns(2)
    with col18:
        st.markdown("**상담 요약**")
        st.text(summary if memo.strip() else "입력 후 요약이 표시된다.")
    with col19:
        st.markdown("**다음 상담 체크리스트**")
        if memo.strip():
            for item in checklist:
                st.write(f"- {item}")
        else:
            st.write("입력 후 체크리스트가 표시된다.")

    st.subheader("8. 상담 메모 키워드 분석")
    keyword_df = get_keyword_counts(memo)
    col20, col21 = st.columns([1, 1])
    with col20:
        st.dataframe(keyword_df, use_container_width=True)
    with col21:
        plot_keyword_chart(keyword_df)

    st.subheader("9. 전체 학생 비교 분석")
    df = load_data()

    if df.empty:
        st.info("아직 저장된 상담 기록이 없습니다.")
    else:
        analysis_df = df.copy()
        numeric_columns = ["gpa", "korean_gpa", "math_gpa", "english_gpa", "inquiry_gpa"]
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
            status_filter = st.multiselect(
                "준비 유형 필터",
                ["학생부종합", "학생부교과", "논술", "정시", "면접 중심", "실기/특기", "기타"]
            )

        filtered_analysis = analysis_df.copy()

        if "gpa" in filtered_analysis.columns:
            filtered_analysis = filtered_analysis[
                (filtered_analysis["gpa"] >= gpa_range[0]) &
                (filtered_analysis["gpa"] <= gpa_range[1])
            ]

        if major_keyword.strip() and "desired_major_1" in filtered_analysis.columns:
            filtered_analysis = filtered_analysis[
                filtered_analysis["desired_major_1"].astype(str).str.contains(major_keyword.strip(), case=False, na=False) |
                filtered_analysis["desired_major_2"].astype(str).str.contains(major_keyword.strip(), case=False, na=False)
            ]

        if university_keyword.strip() and "desired_university_line" in filtered_analysis.columns:
            filtered_analysis = filtered_analysis[
                filtered_analysis["desired_university_line"].astype(str).str.contains(university_keyword.strip(), case=False, na=False)
            ]

        if status_filter and "student_status" in filtered_analysis.columns:
            status_pattern = "|".join(status_filter)
            filtered_analysis = filtered_analysis[
                filtered_analysis["student_status"].astype(str).str.contains(status_pattern, na=False)
            ]

        st.markdown("**필터 결과 요약**")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("해당 학생 수", len(filtered_analysis))
        with m_col2:
            avg_gpa = filtered_analysis["gpa"].mean() if "gpa" in filtered_analysis.columns and not filtered_analysis.empty else None
            st.metric("평균 내신", f"{avg_gpa:.2f}" if avg_gpa is not None else "-")
        with m_col3:
            avg_math = filtered_analysis["math_gpa"].mean() if "math_gpa" in filtered_analysis.columns and not filtered_analysis.empty else None
            st.metric("평균 수학 내신", f"{avg_math:.2f}" if avg_math is not None else "-")

        view_columns = [
            "timestamp", "student_code", "student_status", "desired_university_line",
            "desired_major_1", "desired_major_2", "priority_type", "gpa", "math_gpa",
            "mock_korean", "mock_math", "mock_english", "mock_inquiry",
            "academic_level", "career_level", "community_level",
            "upper_universities", "target_universities", "safe_universities", "next_plan"
        ]
        existing_view_columns = [col for col in view_columns if col in filtered_analysis.columns]
        st.dataframe(filtered_analysis[existing_view_columns], use_container_width=True)

        st.markdown("**그룹별 간단 비교**")
        group_col1, group_col2 = st.columns(2)
        with group_col1:
            if "desired_major_1" in filtered_analysis.columns and not filtered_analysis.empty:
                major_counts = filtered_analysis["desired_major_1"].value_counts().reset_index()
                major_counts.columns = ["희망 전공 1지망", "학생 수"]
                st.dataframe(major_counts, use_container_width=True)
        with group_col2:
            if "desired_university_line" in filtered_analysis.columns and not filtered_analysis.empty:
                line_counts = filtered_analysis["desired_university_line"].value_counts().reset_index()
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
