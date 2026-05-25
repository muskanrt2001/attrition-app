
import streamlit as st
import pandas as pd
import numpy as np
import joblib, shap, matplotlib.pyplot as plt, warnings
warnings.filterwarnings("ignore")

@st.cache_resource
def load_artifacts():
    model      = joblib.load("best_attrition_model.pkl")
    explainer  = joblib.load("shap_explainer.pkl")
    feat_names = joblib.load("feature_names.pkl")
    meta       = joblib.load("model_meta.pkl")
    return model, explainer, feat_names, meta

model, explainer, feat_names, meta = load_artifacts()
NUM_COLS   = meta["num_cols"]
CAT_COLS   = meta["cat_cols"]
MODEL_NAME = meta["best_model_name"]

st.set_page_config(page_title="Attrition Predictor", page_icon="👥", layout="wide")
st.markdown('''<style>
.main-title{font-size:2.2rem;font-weight:700;color:#1a1a2e}
.sub-title{font-size:1rem;color:#555;margin-bottom:1.5rem}
.risk-low{background:#d5f5e3;border-radius:10px;padding:18px;text-align:center}
.risk-medium{background:#fef9e7;border-radius:10px;padding:18px;text-align:center}
.risk-high{background:#fadbd8;border-radius:10px;padding:18px;text-align:center}
.stButton>button{width:100%;background:#2c3e50;color:white;border-radius:8px;
                 padding:.6rem;font-size:1rem;font-weight:600}
.stButton>button:hover{background:#3498db}
</style>''', unsafe_allow_html=True)

st.markdown('<p class="main-title">👥 Employee Attrition Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-powered HR Analytics — Assess attrition risk instantly.</p>', unsafe_allow_html=True)
st.caption(f"Model in use: **{MODEL_NAME}**")
st.divider()

def ag(a):
    if a<25:return"<25"
    elif a<35:return"25-35"
    elif a<45:return"35-45"
    elif a<55:return"45-55"
    else:return"55+"

def tg(t):
    if t<1:return"<1yr"
    elif t<5:return"1-5yr"
    elif t<10:return"5-10yr"
    elif t<20:return"10-20yr"
    else:return"20yr+"

def sen(b):
    return{"Non-Officer":1,"Junior Management":2,"Middle Management":3,
           "Upper Middle Management":4,"Top Management":5,"Other":1}.get(b,1)

def risk_label(p):
    if p<.30:return"Low Risk","risk-low","🟢"
    elif p<.60:return"Medium Risk","risk-medium","🟡"
    else:return"High Risk","risk-high","🔴"

def get_recs(risk,rel,rpts,esg):
    r=[]
    if risk=="High Risk":
        r+=["🚨 **Immediate manager check-in** — schedule 1:1 within 1 week.",
            "💰 **Compensation review** — benchmark against market rates.",
            "🎯 **Career path discussion** — define clear promotion milestones.",
            "🏆 **Recognition program** — nominate for spot award.",
            "📋 **Stay interview** — understand key pain points."]
    elif risk=="Medium Risk":
        r+=["📈 **Learning & development** — enrol in a skill-building course.",
            "🤝 **Mentorship pairing** — connect with a senior buddy.",
            "⚖️ **Work-life balance review** — assess workload and overtime.",
            "🏢 **Team engagement** — include in next offsite."]
    else:
        r+=["✅ **Maintain engagement** — continue regular feedback cycles.",
            "🌱 **Growth opportunities** — offer stretch assignments.",
            "🎉 **Celebrate milestones** — acknowledge tenure anniversaries."]
    if rel!="Yes": r.append("📍 **Relocation support** — explore remote/hybrid options.")
    if rpts==0 and sen(esg)>=2: r.append("👑 **Leadership program** — consider management fast-track.")
    return r

st.sidebar.header("🧾 Employee Details")
with st.sidebar:
    age=st.slider("Age",18,75,35)
    tenure=st.slider("Tenure (yrs)",0,40,5)
    dr=st.number_input("Direct Reports",0,50,0,step=1)
    fw=st.number_input("Final Weight (kg)",30.0,150.0,68.3,step=0.5)
    fh=st.number_input("Final Height (cm)",120.0,220.0,166.5,step=0.5)
    dept=st.selectbox("Department",sorted(["Engineering","Process","Administration","Accounts","Distillery",
        "Human Resources","Information Technology","Legal","Marketing","Operations",
        "Production","Quality","R&D","Sales","Security","Safety","Supply Chain","Other"]))
    loc=st.selectbox("Location",["Sugar&Distillery Unit-Hariawan (Dsha)",
        "Sugar&Distillery Unit-Ajbapur (Asc)","Sugar Unit - Loni (Dslo)",
        "Sugar Unit - Rupapur (Rusc)","Corporate Sugar (Csug)","Corporate Office (Corp)",
        "Shriram Bio Enchem Limited (Dsbe)","Other"])
    esg=st.selectbox("ESG Band",["Non-Officer","Junior Management","Middle Management",
        "Upper Middle Management","Top Management","Other"])
    esg_lv=st.text_input("ESG Level","E1")
    role=st.selectbox("Role Name",["Engineer","Manager","Officer","Supervisor","Worker",
        "Technician","Executive","Analyst","Director","Other"])
    ms=st.selectbox("Marital Status",["Married","Single","Other","Unknown"])
    gen=st.selectbox("Gender",[1,0],format_func=lambda x:"Male" if x==1 else "Female")
    rel=st.selectbox("Willing to Relocate",["Yes","No","Unknown"])
    relig=st.selectbox("Religion",["Hindu","Muslim","Christian","Sikh","Other","Unknown"])
    caste=st.selectbox("Caste Category",["General","Obc","Sc","St","Other","Unknown"])
    bg=st.selectbox("Blood Group",["A+","A-","B+","B-","Ab+","Ab-","O+","O-","Unknown"])
    edu=st.slider("Education Duration (yrs)",0.0,15.0,3.0,step=0.5)

if st.sidebar.button("🔍  Predict Attrition Risk",use_container_width=True):
    bmi=max(10.0,min(60.0,fw/((fh/100)**2)))
    inp=pd.DataFrame([{
        "Direct Reports":dr,"Department":dept,"Location":loc,"Gender":gen,
        "Blood Group":bg,"Willing to Relocate":rel,"Role Name":role,
        "age":age,"tenure":tenure,"Religion":relig,"final weights":fw,"final height":fh,
        "Caste Category":caste,"ESG Level":esg_lv,"ESG Band":esg,"Marital Status":ms,
        "bmi":bmi,"edu_duration_years":edu,"age_group":ag(age),"tenure_group":tg(tenure),
        "is_leader":int(dr>0),"relocation_flexible":int(rel=="Yes"),"seniority_score":sen(esg),
    }])
    prob=model.predict_proba(inp)[0]
    pl,ps=prob[0],prob[1]
    risk,badge,emoji=risk_label(pl)

    t1,t2,t3=st.tabs(["📊 Prediction","🔬 SHAP","💡 Recommendations"])
    with t1:
        c1,c2,c3=st.columns(3)
        c1.metric("Attrition Probability",f"{pl:.1%}")
        c2.metric("Retention Probability",f"{ps:.1%}")
        with c3:
            st.markdown(f'<div class="{badge}"><h3>{emoji} {risk}</h3><p style="margin:0;font-size:.85rem">Confidence: {max(pl,ps):.1%}</p></div>',unsafe_allow_html=True)
        st.divider()
        fig,ax=plt.subplots(figsize=(7,1.2))
        ax.barh([""],[pl],color="#e74c3c",height=0.4)
        ax.barh([""],[ps],left=[pl],color="#2ecc71",height=0.4)
        ax.axvline(0.30,color="orange",ls="--",lw=1.2,label="30%")
        ax.axvline(0.60,color="red",ls="--",lw=1.2,label="60%")
        ax.set_xlim(0,1);ax.set_xlabel("Probability")
        ax.legend(fontsize=7,loc="lower right")
        ax.set_title("Attrition vs Retention Probability")
        plt.tight_layout();st.pyplot(fig);plt.close()
        st.markdown(f'''| Field | Value |
|--|--|
| Age | {age} |
| Tenure | {tenure} yrs |
| Department | {dept} |
| ESG Band | {esg} |
| BMI | {bmi:.1f} |
| Direct Reports | {dr} |''')
    with t2:
        try:
            pre=model.named_steps["pre"]
            Xt=pre.transform(inp)
            sv=explainer.shap_values(Xt)
            if isinstance(sv,list):sv=sv[1]
            sv_row=sv[0] if sv.ndim==2 else sv
            s=pd.Series(sv_row,index=feat_names)
            top=s.abs().sort_values(ascending=False).head(15)
            cols=["#e74c3c" if s[f]>0 else "#2ecc71" for f in top.index[::-1]]
            fig,ax=plt.subplots(figsize=(9,5))
            ax.barh(top.index[::-1],s[top.index[::-1]].values,color=cols,edgecolor="white")
            ax.axvline(0,color="black",lw=0.8)
            ax.set_title('''Top 15 SHAP Features
(Red=↑ risk, Green=↓ risk)''',fontsize=11)
            ax.set_xlabel("SHAP Value")
            plt.tight_layout();st.pyplot(fig);plt.close()
            st.markdown("**Top 5 drivers:**")
            for i,(f,_) in enumerate(top.head(5).items(),1):
                d="↑ increases" if s[f]>0 else "↓ decreases"
                st.write(f"{i}. `{f}` — {d} risk ({s[f]:.4f})")
        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")
    with t3:
        recs=get_recs(risk,rel,dr,esg)
        st.markdown(f"#### {emoji} {risk}")
        for r in recs:st.markdown(r)
        st.divider()
        st.markdown("#### 📅 30-Day Action Plan")
        if risk=="High Risk":
            plan=[("Week 1","1:1 + stay interview"),("Week 2","Comp benchmarking"),
                  ("Week 3","Career roadmap"),("Week 4","Recognition nomination")]
        elif risk=="Medium Risk":
            plan=[("Week 1","Enrol in L&D"),("Week 2","Assign mentor"),
                  ("Week 3","Workload audit"),("Week 4","Team activity")]
        else:
            plan=[("Week 1","Appreciation note"),("Week 2","Stretch assignment"),
                  ("Week 3","L&D goals"),("Week 4","Milestone ack.")]
        for w,a in plan:st.markdown(f"- **{w}**: {a}")
    st.sidebar.success(f"{emoji} Done!")
else:
    st.info("👈  Fill in employee details in the sidebar and click **Predict Attrition Risk**.")
