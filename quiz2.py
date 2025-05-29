import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model   import LogisticRegression
from sklearn.metrics        import accuracy_score, confusion_matrix


# 3. 데이터 전처리
df = pd.read_csv('한국_기업문화_HR_데이터셋_샘플.csv')

# 결측치 확인 -> 결측치가 따로 없어서 필요한 전처리만
print("결측치 현황:\n", df.isna().sum())

# 이직여부(target) 이진화 
df['이직여부'] = df['이직여부'].map({'Yes':1, 'No':0})
df = pd.get_dummies(df, columns=['야근여부'], drop_first=True)
# 4. 피처 선택 (근거도)
#  Age: 나이↑ → 이직율↓ 
#  월급여: 보상↑ → 이직율↓ 
#  집까지거리: 거리↑ → 이직율↑ 
#  업무만족도: 만족↑ → 이직율↓ 
#  야근여부_Yes: 잔업 시 이직율↑ 
#  총경력: 경력↑ → 이직율↑ 
#  현회사근속년수: 근속↑ → 이직율↓ 
#  근무환경만족도: 만족↑ → 이직율↓ 
#  워라밸: 균형↑ → 이직율↓ 
features = [
    'Age',             
    '월급여',           
    '집까지거리',        
    '업무만족도',
    '야근여부_Yes',
    '총경력',
    '현회사근속년수',
    '근무환경만족도',
    '워라밸'
]

# 5. 모델 훈련

# 학습/테스트 분할 (8:2)
X = df[features]
y = df['이직여부']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=1000)  
model.fit(X_train, y_train)                

y_pred = model.predict(X_test)             
proba  = model.predict_proba(X_test)[:,1]  
# 6. 성능 검증 해석
# 정확도: 0.870 → 전체 샘플 중 87.0%를 올바르게 분류
# 혼동행렬: [[TN=167, FP=1], [FN=25, TP=7]]
# TN 높음: 잔류(No) 예측이 대부분 정확
# TP 낮음: 이직(Yes) 예측은 일부 놓침

acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)
print(f"정확도: {acc:.3f}")
print("혼동행렬:\n", cm)
print()
# 7. 예측 결과 분석
n_yes = (y_pred == 1).sum()
print(f"▶ 테스트셋에서 이직여부=Yes로 예측된 직원 수: {n_yes}")
print()
df_test = X_test.copy()
df_test['y_true']   = y_test.values
df_test['y_pred']   = y_pred
df_test['prob_yes'] = proba

top5 = df_test.sort_values('prob_yes', ascending=False).head(5)
print("▶ 이직 가능성 상위 5명:\n", top5)
print()
# 8. 신입사원 예측
# 신규 예측 결과 [1, 0, 1]
# 1번·3번 사원: 이직 가능성 높음(관리 필요)
# 2번 사원: 이직 가능성 낮음

new_hires = [
    {
        "Age": 29, "집까지거리":5, "월급여":2800,
        "업무만족도":2, "근무환경만족도":2, "워라밸":2,
        "총경력":4, "현회사근속년수":1,
        "야근여부": "Yes"
    },
    {
        "Age": 42, "집까지거리":10, "월급여":5200,
        "업무만족도":4, "근무환경만족도":3, "워라밸":3,
        "총경력":18, "현회사근속년수":7,
        "야근여부": "No"
    },
    {
        "Age": 35, "집까지거리":2, "월급여":3300,
        "업무만족도":1, "근무환경만족도":1, "워라밸":2,
        "총경력":10, "현회사근속년수":2,
        "야근여부": "Yes"
    }
]

df_new = pd.DataFrame(new_hires)

df_new['야근여부_Yes'] = df_new['야근여부'].map({'Yes':1,'No':0})

X_new = df_new[features]

new_preds = model.predict(X_new)
print("[신규 입사자 이직 예측 결과 (0=No, 1=Yes)]:", list(new_preds))


# 9. 영향력 큰 Top3 피처 파악 
# 야근여부_Yes (coef=+1.349): 야근 시 이직 확률 크게 상승
# 업무만족도   (coef=–0.396): 만족도↑ → 이직 확률 감소
# 근무환경만족도 (coef=–0.321): 환경 만족↑ → 이직 확률 감소
coefs = pd.Series(model.coef_[0], index=features)
top3 = coefs.abs().sort_values(ascending=False).head(3)
print("[이직 예측에 가장 큰 영향을 준 Top3 피처 (절대값 순)]:")
print(top3)
print()

