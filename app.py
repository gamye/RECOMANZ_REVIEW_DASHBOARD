import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# ----------------------------------
# 페이지 기본 설정
# ----------------------------------
st.set_page_config(
    page_title="리코맨즈 리뷰 분석 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------
# 데이터 로딩 및 전처리 (캐싱 활용)
# ----------------------------------
@st.cache_data
def load_data(filepath):
    """CSV 데이터를 로드하고 전처리합니다."""
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    except Exception as e:
        st.error(f"데이터 로딩 중 오류 발생: {e}")
        return pd.DataFrame()
    df['리뷰작성일시'] = pd.to_datetime(df['리뷰작성일시'], errors='coerce')
    df.dropna(subset=['리뷰작성일시'], inplace=True)
    df['리뷰내용'] = df['리뷰내용'].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
    return df

# ----------------------------------
# 분석 함수
# ----------------------------------
okt = Okt()

# --- 불용어 목록 정의 ---
stopwords = [
    '하다', '있다', '되다', '그렇다', '않다', '없다', '같다', '보이다', '들다', '나다', '오다', '가다', '온건',
    '이', '그', '저', '것', '수', '등', '좀', '잘', '더', '많이', '조금', '정말', '진짜', '너무', '만족하다', '편이',
    '또', '다시', '생각', '느낌', '사용', '주문', '구매', '제품', '상품', '네이버', '리뷰', '사진', '확인', '중이', '이바',
    '하나', '저희', '이번', '그냥', '한번', '같아요', '입니다', '있어요', '해요', '되요', '네요', '하니', '차고', '주신',
    '하고', '해서', '하면', '으로', '으로도', '에게', '지만', '는데', '습니다', '하게', '으니', '문해', '약간', '기도',
    'ㅎㅎ', 'ㅋㅋ', 'ㅠㅠ', '^^', ':)','~','!','.', '페이', '등록', '등록된', '아니다', '부분', '이다', '하루', '물건', '드네'
]

def preprocess_text(text):
    """텍스트 정제 함수: 형태소 분석 및 불용어 처리"""
    if not isinstance(text, str):
        return ""
    morphs = okt.pos(text, stem=True, norm=True)
    words = [word for word, pos in morphs if pos in ['Noun', 'Adjective']]
    meaningful_words = [w for w in words if len(w) > 1 and w not in stopwords]
    return ' '.join(meaningful_words)

# @st.cache_data # 캐싱 제거
def analyze_all_keywords_tfidf(_df):
    """데이터프레임 전체에 대해 TF-IDF를 계산하고 단어별 긍정/부정성 점수를 반환합니다."""
    pos_reviews_df = _df[_df['리뷰평점'].isin([4, 5])]
    neg_reviews_df = _df[_df['리뷰평점'].isin([1, 2])]

    if pos_reviews_df.empty or neg_reviews_df.empty:
        return pd.DataFrame()

    pos_corpus = pos_reviews_df['리뷰내용'].apply(preprocess_text)
    neg_corpus = neg_reviews_df['리뷰내용'].apply(preprocess_text)
    
    vectorizer = TfidfVectorizer(max_features=1000)
    all_corpus = pd.concat([pos_corpus, neg_corpus])
    if all_corpus.empty or all_corpus.str.strip().eq('').all():
        return pd.DataFrame()
        
    vectorizer.fit(all_corpus)
    
    pos_tfidf_matrix = vectorizer.transform(pos_corpus)
    neg_tfidf_matrix = vectorizer.transform(neg_corpus)
    
    words = vectorizer.get_feature_names_out()
    pos_scores = np.array(pos_tfidf_matrix.mean(axis=0)).flatten()
    neg_scores = np.array(neg_tfidf_matrix.mean(axis=0)).flatten()

    tfidf_df = pd.DataFrame({'단어': words, '긍정점수': pos_scores, '부정점수': neg_scores})
    tfidf_df['긍정성'] = tfidf_df['긍정점수'] - tfidf_df['부정점수']
    
    return tfidf_df

def get_product_negativity_score(review_series, keyword_scores):
    """상품 리뷰들의 부정성 점수 평균을 계산합니다."""
    total_score = 0
    word_count = 0
    neg_keyword_df = keyword_scores[keyword_scores['긍정성'] < 0]
    
    for review in review_series:
        processed_review = preprocess_text(review).split()
        for word in processed_review:
            if word in neg_keyword_df['단어'].values:
                score = neg_keyword_df[neg_keyword_df['단어'] == word]['긍정성'].iloc[0] * -1
                total_score += score
                word_count += 1
    
    return total_score / word_count if word_count > 0 else 0

def highlight_keywords(text, pos_keywords, neg_keywords):
    """리뷰 텍스트 내 키워드를 HTML로 하이라이팅합니다."""
    if not isinstance(text, str):
        return ""
    highlighted_text = text
    for word in sorted(pos_keywords, key=len, reverse=True):
        highlighted_text = re.sub(f'({re.escape(word)})', r'<span style="color:blue; font-weight:bold;">\1</span>', highlighted_text)
    for word in sorted(neg_keywords, key=len, reverse=True):
         highlighted_text = re.sub(f'({re.escape(word)})', r'<span style="color:red; font-weight:bold;">\1</span>', highlighted_text)
    return highlighted_text

# ----------------------------------
# 메인 대시보드 UI
# ----------------------------------
df_reviews = load_data('240611-250611_Quick_Review_Filter.csv')

if not df_reviews.empty:
    st.title("리코맨즈 리뷰 분석 대시보드 (TF-IDF 기반)")
    st.markdown("---")

    # --- 사이드바 ---
    st.sidebar.header("🗓️ 기간 필터")
    min_date = df_reviews['리뷰작성일시'].min().date()
    max_date = df_reviews['리뷰작성일시'].max().date()
    start_date, end_date = st.sidebar.date_input("조회할 기간을 선택하세요.", (min_date, max_date), min_value=min_date, max_value=max_date, format="YYYY-MM-DD")

    filtered_df = df_reviews[(df_reviews['리뷰작성일시'].dt.date >= start_date) & (df_reviews['리뷰작성일시'].dt.date <= end_date)]

    if filtered_df.empty:
        st.warning("선택된 기간에 해당하는 리뷰 데이터가 없습니다.")
    else:
        keyword_scores_df = analyze_all_keywords_tfidf(filtered_df)

        st.header("주요 지표 및 트렌드")
        avg_rating = filtered_df['리뷰평점'].mean()
        total_avg_rating = df_reviews['리뷰평점'].mean()
        st.metric(label="기간 내 전체 평균 평점", value=f"{avg_rating:.2f} 점", delta=f"{avg_rating - total_avg_rating:.2f} (전체 평균 대비)")

        time_diff = end_date - start_date
        
        if time_diff.days > 60:
            period_name = "월별"
            agg_df = filtered_df.set_index('리뷰작성일시').groupby(pd.Grouper(freq='M')).agg(리뷰개수=('리뷰번호', 'count'), 평균평점=('리뷰평점', 'mean')).reset_index()
            agg_df = agg_df[agg_df['리뷰개수'] > 0]
            agg_df['기간'] = agg_df['리뷰작성일시'].dt.strftime('%Y-%m')
        else:
            period_name = "일별"
            agg_df = filtered_df.set_index('리뷰작성일시').groupby(pd.Grouper(freq='D')).agg(리뷰개수=('리뷰번호', 'count'), 평균평점=('리뷰평점', 'mean')).reset_index()
            agg_df = agg_df[agg_df['리뷰개수'] > 0]
            agg_df['기간'] = agg_df['리뷰작성일시'].dt.strftime('%m-%d')

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Bar(x=agg_df['기간'], y=agg_df['리뷰개수'], name='리뷰 개수', marker_color='skyblue', text=agg_df['리뷰개수'], textposition='outside', textfont=dict(size=10)), secondary_y=False)
        fig.add_trace(go.Scatter(x=agg_df['기간'], y=agg_df['평균평점'], name='평균 평점', mode='lines+markers', marker_color='darkorange'), secondary_y=True)

        fig.update_layout(title_text=f'<b>{period_name} 리뷰 개수 및 평균 평점 트렌드</b>', xaxis_title=period_name, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(gridcolor='rgba(0,0,0,0)'), yaxis2=dict(gridcolor='rgba(0,0,0,0)'), xaxis=dict(showgrid=False, type='category', tickangle=-45), uniformtext_minsize=8, uniformtext_mode='hide')
        fig.update_yaxes(title_text="<b>리뷰 개수</b> (건)", secondary_y=False, showgrid=False, zeroline=False)
        fig.update_yaxes(title_text="<b>평균 평점</b> (점)", secondary_y=True, range=[1, 5], showgrid=False, zeroline=False)

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")

        st.header("긍정 & 부정 키워드 분석")
        with st.expander("TF-IDF 기반 감성분석 설명 보기"):
            st.markdown("""기존의 고정된 감성 사전을 사용하는 대신, **실제 리뷰 데이터 자체를 학습하여** 긍정/부정 키워드를 추출하는 데이터 기반 분석 방식입니다.

        1. 데이터 그룹화: 먼저 리뷰를 긍정 그룹(4-5점)과 부정 그룹(1-2점)으로 나눕니다.
            
        2. TF-IDF 점수 계산: 각 그룹 내에서 특정 단어가 얼마나 자주 등장하고(TF), 다른 그룹에서는 얼마나 드물게 나타나는지(IDF)를 종합하여 단어의 중요도(TF-IDF 점수)를 계산합니다.
            
        3. '긍정성' 점수화: 한 단어의 긍정 그룹 TF-IDF 평균 점수에서 부정 그룹 TF-IDF 평균 점수를 뺀 값을 긍정성 점수로 정의합니다.
            - 이 점수가 높을수록 해당 단어는 긍정적인 맥락에서 더 중요하게 사용되었음을 의미합니다. (예: `재구매`, `고급`)
            - 이 점수가 낮을수록(음수일수록) 부정적인 맥락에서 더 중요하게 사용되었음을 의미합니다. (예: `분실`, `변색`, `자석`)
            
        이 방식을 통해, '가격 대비'(부정)나 '선물용'(긍정)처럼 우리 쇼핑몰의 고유한 맥락이 담긴 핵심 키워드를 효과적으로 발견할 수 있습니다.""")

        if not keyword_scores_df.empty:
            pos_keywords_df = keyword_scores_df[keyword_scores_df['긍정점수'] > 0.01].sort_values(by='긍정성', ascending=False).head(10)
            neg_keywords_df = keyword_scores_df[keyword_scores_df['부정점수'] > 0.01].sort_values(by='긍정성', ascending=True).head(10)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("👍 긍정 키워드 TOP 10")
                st.dataframe(pos_keywords_df[['단어', '긍정성']].rename(columns={'단어': '키워드', '긍정성': '긍정성 점수'}), use_container_width=True, hide_index=True)
            with col2:
                st.subheader("👎 부정 키워드 TOP 10")
                st.dataframe(neg_keywords_df[['단어', '긍정성']].rename(columns={'단어': '키워드', '긍정성': '긍정성 점수'}), use_container_width=True, hide_index=True)
        else:
            st.info("키워드 분석을 위한 데이터가 부족합니다 (긍정/부정 리뷰가 모두 필요합니다).")

        st.markdown("---")

        st.header("개선 필요 상품 분석")
        st.info("개선이 필요한 상품 목록입니다. **표의 행을 클릭**하면 아래에서 상세 리뷰를 확인할 수 있습니다.")

        with st.expander("개선 필요 점수 산정 방식 설명 보기"):
            st.markdown("""**개선 필요 점수**는 상품별 개선 우선순위를 정하기 위한 지표입니다. 점수가 높을수록 많은 고객이 불편을 겪고 있어 우선적인 개선이 필요함을 의미합니다.

     1.  TF-IDF 부정성 영향 (가중치 20%): 해당 상품의 리뷰에 등장한 부정 키워드들의 TF-IDF 기반 부정성 점수를 평균내어 계산합니다. 단순히 부정 단어의 빈도가 아닌, '얼마나 강한' 부정 표현이 쓰였는지를 측정합니다.
            
     2.  리뷰 수를 반영한 낮은 평점 영향 (가중치 70%): 인기 상품(리뷰 수가 많은 상품)의 낮은 평점에 더 큰 페널티를 부여합니다. 많은 고객이 경험하는 대표적인 불만일수록 개선이 시급하기 때문입니다.
            
     3.  평점 변동성 영향 (가중치 10%): 평점의 표준편차를 사용하여 평가의 일관성을 측정합니다. 호불호가 심하게 갈리는 상품을 식별합니다.""")
            
        with st.spinner("개선 필요 상품 분석 중..."):
            product_stats = filtered_df.groupby(['상품명', '상품코드']).agg(리뷰수=('리뷰번호', 'count'), 평균평점=('리뷰평점', 'mean'), 평점표준편차=('리뷰평점', 'std')).reset_index()
            product_stats['평점표준편차'].fillna(0, inplace=True)

            if not keyword_scores_df.empty:
                product_stats['부정성점수'] = product_stats.apply(lambda row: get_product_negativity_score(filtered_df[filtered_df['상품코드'] == row['상품코드']]['리뷰내용'], keyword_scores_df), axis=1)
                max_log_reviews = np.log1p(product_stats['리뷰수'].max())
                product_stats['리뷰수가중치'] = np.log1p(product_stats['리뷰수']) / max_log_reviews if max_log_reviews > 0 else 0
                max_neg_score = product_stats['부정성점수'].max()
                negativity_norm = product_stats['부정성점수'] / max_neg_score if max_neg_score > 0 else 0
                low_rating_norm = ((5 - product_stats['평균평점']) / 4) * product_stats['리뷰수가중치']
                std_dev_norm = product_stats['평점표준편차'] / 2
                w_neg, w_rating, w_std = 0.2, 0.7, 0.1
                product_stats['개선필요점수'] = (negativity_norm * w_neg + low_rating_norm * w_rating + std_dev_norm * w_std) * 100
            else:
                product_stats['부정성점수'] = 0
                product_stats['개선필요점수'] = 0

            improvement_df = product_stats.sort_values(by='개선필요점수', ascending=False)
            display_cols = ['상품명', '리뷰수', '평균평점', '부정성점수', '개선필요점수']
            improvement_df_display = improvement_df[display_cols].reset_index(drop=True)
            improvement_df_display['평균평점'] = improvement_df_display['평균평점'].round(1)
            improvement_df_display['부정성점수'] = improvement_df_display['부정성점수'].round(3)
            improvement_df_display['개선필요점수'] = improvement_df_display['개선필요점수'].round(1)

            gb = GridOptionsBuilder.from_dataframe(improvement_df_display)
            gb.configure_selection('single', use_checkbox=False)
            
            # domLayout='autoHeight' 옵션을 제거합니다.
            grid_options = gb.build()

            grid_response = AgGrid(
                improvement_df_display,
                gridOptions=grid_options,
                height=400,  # 그리드의 높이를 400px로 고정하여 스크롤바 생성
                update_mode=GridUpdateMode.MODEL_CHANGED,
                allow_unsafe_jscode=True,
                theme='streamlit',
                key='product_grid'
            )
            
            selected_rows = grid_response['selected_rows']
            
            if selected_rows is not None and not selected_rows.empty:
                selected_product_name = selected_rows.iloc[0]['상품명']
            
                st.markdown("---")
                st.header(f"🔍 '{selected_product_name}' 상세 리뷰")
                
                detail_df = filtered_df[filtered_df['상품명'] == selected_product_name]
                
                if not keyword_scores_df.empty:
                    pos_k_list = keyword_scores_df[keyword_scores_df['긍정성'] > 0]['단어'].tolist()
                    neg_k_list = keyword_scores_df[keyword_scores_df['긍정성'] < 0]['단어'].tolist()
                else:
                    pos_k_list, neg_k_list = [], []
                
                for _, row in detail_df.iterrows():
                    rating_stars = "⭐" * int(row['리뷰평점']) + "☆" * (5 - int(row['리뷰평점']))
                    st.markdown(f"**평점:** {rating_stars} (`{row['리뷰평점']}`점)")
                    
                    highlighted_review = highlight_keywords(row['리뷰내용'], pos_k_list, neg_k_list)
                    st.markdown(f"> {highlighted_review}", unsafe_allow_html=True)
                    st.markdown("---")