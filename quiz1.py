import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 불러오기 및 전처리
# [1-1] 데이터프레임으로 불러오기
df = pd.read_csv('201906.csv', encoding='utf-8')

# [1-2] 필요 컬럼만 선택 & 컬럼명 변경
df = df[['날짜', '측정소명', '미세먼지', '초미세먼지']]
df = df.rename(columns={
    '날짜': 'date',
    '측정소명': 'district',
    '미세먼지': 'PM10',
    '초미세먼지': 'PM2.5'
})

# [1-3] 첫 행의 요약(전체) 제거
df = df[df['date'] != '전체']

# [1-4] 자료형 변환 및 결측치 제거
df['date']   = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['PM10']   = pd.to_numeric(df['PM10'], errors='coerce')
df['PM2.5']  = pd.to_numeric(df['PM2.5'], errors='coerce')
df = df.dropna(subset=['date','district','PM10','PM2.5'])


# 2. 파생변수 만들기
# [2-1] month, day 파생변수 생성
df['month'] = df['date'].dt.month
df['day']   = df['date'].dt.day

# [2-2] 계절(season) 변수 생성: month 기준으로 spring/summer/autumn/winter
def get_season(m):
    if m in [3, 4, 5]:
        return 'spring'
    elif m in [6, 7, 8]:
        return 'summer'
    elif m in [9, 10, 11]:
        return 'autumn'
    else:
        return 'winter'

df['season'] = df['month'].apply(get_season)


# 3. 전처리 완료 데이터 확인 및 저장
# [3-1] 최종 분석 대상 데이터 확인
print(df.head())
print()

# [3-2] 'card_output.csv'로 저장
df.to_csv('card_output.csv', index=False)


# 4. 연간(전체) 미세먼지 평균 구하기
# [4-1] 전체 데이터 기준 PM10 평균
avg_pm10 = df['PM10'].mean()
print(f'전체 PM10 평균: {avg_pm10:.2f} μg/m³')
print()


# 5. 미세먼지 최댓값 날짜 및 지역 확인
# [5-1] PM10 최댓값 발생한 행을 한 개 꺼내서 날짜와 지역을 함께 출력
max_row = df.loc[df['PM10'] == df['PM10'].max()].iloc[0]
print(f"PM10 최댓값 {max_row['PM10']} μg/m³ 발생일: {max_row['date'].date()}, 지역: {max_row['district']}")
print()


# 6. 구별 PM10 평균 비교
# [6-1] 각 구별 PM10 평균 계산
district_avg = (
    df.groupby('district')['PM10']
      .mean()
      .reset_index(name='avg_PM10')
)

# [6-2] 상위 5개 구만 출력
top5_district = district_avg.sort_values('avg_PM10', ascending=False).head(5)
print('\n구별 PM10 평균 상위 5개')
print(top5_district)
print()


# 7. 계절별 PM10/PM2.5 평균 비교
# [7-1] 계절별 평균 PM10, PM2.5 동시 계산
season_avg = (
    df.groupby('season')[['PM10','PM2.5']]
      .mean()
      .reset_index()
      .rename(columns={'PM10':'avg_PM10','PM2.5':'avg_PM2.5'})
)

# [7-2] avg_PM10 기준 오름차순 정렬
season_avg = season_avg.sort_values('avg_PM10', ascending=True)
print('\n계절별 미세먼지 평균')
print(season_avg)
print()


# 8. PM10 등급(pm_grade) 분류 및 분포 확인
# [8-1] PM10 값을 기준으로 등급 분류 (good/normal/bad/worse)
def grade_pm10(x):
    if x <= 30:
        return 'good'
    elif x <= 80:
        return 'normal'
    elif x <= 150:
        return 'bad'
    else:
        return 'worse'

df['pm_grade'] = df['PM10'].apply(grade_pm10)

# [8-2] 등급별 빈도(n), 비율(pct) 계산 — 문자열/타입 문제 방지
grade_dist = (
    df['pm_grade']
      .value_counts()
      .rename_axis('pm_grade')
      .reset_index(name='n')
)
grade_dist['pct'] = grade_dist['n'] / grade_dist['n'].sum() * 100
print('\nPM10 등급 분포')
print(grade_dist)
print()


# 9. 구별 good 등급 비율 상위 5개 구 추출
# [9-1] 구별 'good' 빈도 및 전체 대비 비율 계산
good_cnt = (
    df[df['pm_grade']=='good']
      .groupby('district')
      .size()
      .reset_index(name='good_n')
)
total_cnt = (
    df.groupby('district')
      .size()
      .reset_index(name='total_n')
)
good_ratio = pd.merge(good_cnt, total_cnt, on='district')
good_ratio['pct'] = good_ratio['good_n'] / good_ratio['total_n'] * 100

# [9-2] pct 기준 내림차순 정렬 후 상위 5개 구만 출력
top5_good = good_ratio.sort_values('pct', ascending=False).head(5)
print('\n구별 Good 등급 비율 상위 5개 구')
print(top5_good[['district','good_n','total_n','pct']])
print()


# 10. 1년간 일별 미세먼지 추이 그래프
plt.figure(figsize=(12,6))
# marker 없이 선만
plt.plot(df['date'], df['PM10'], linestyle='-')
plt.xlabel('Date')
plt.ylabel('PM10 (μg/m³)')
plt.title('Daily Trend of PM10 in Seoul, 2019')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show


# 11. 계절별 PM10 등급 비율 그래프
season_grade = (
    df.groupby(['season','pm_grade'])
      .size()
      .reset_index(name='count')
)

season_grade['pct'] = (
    season_grade['count']
    / season_grade.groupby('season')['count']
                    .transform('sum')
    * 100
)

plt.figure(figsize=(8,6))
sns.barplot(
    data=season_grade,
    x='season',
    y='pct',
    hue='pm_grade'
)
plt.title('Seasonal Distribution of PM10 Grades in Seoul, 2019')
plt.ylabel('Percentage (%)')
plt.xlabel('Season')
plt.legend(title='PM10 Grade')
plt.tight_layout()
plt.show()