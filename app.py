import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.datasets import load_diabetes, load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

st.set_page_config(page_title="DataSphere", page_icon="🌐", layout="wide")
st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#eef6ff,#f7fbff,#e7f4fb)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#071a28,#0c2936)}
[data-testid="stSidebar"] *{color:#f3f8fc!important}
.hero{padding:2rem;border-radius:20px;color:white;background:radial-gradient(circle at 80% 20%,#087f9d 0,transparent 28%),linear-gradient(120deg,#061b32,#0a3150 65%,#0c596b)}
.hero h1{font-size:2.8rem;margin-bottom:.3rem}.hero p{font-size:1.05rem;max-width:850px}
div[data-testid="stMetric"]{background:#ffffffd9;border:1px solid #d8e7f2;padding:12px;border-radius:13px}
</style>""", unsafe_allow_html=True)

@st.cache_data
def dataset(name):
    if name == "Diabetes (Regression)":
        b=load_diabetes(as_frame=True); df=b.frame.copy(); target="target"
        desc="Dataset diabetes untuk regresi." 
    elif name == "Iris (Classification / Clustering)":
        b=load_iris(as_frame=True); df=b.frame.copy(); target="target"
        desc="Dataset bunga Iris dengan empat fitur dan tiga kelas."
    elif name == "Wine (Classification)":
        b=load_wine(as_frame=True); df=b.frame.copy(); target="target"
        desc="Dataset kimia wine untuk klasifikasi."
    else:
        b=load_breast_cancer(as_frame=True); df=b.frame.copy(); target="target"
        desc="Dataset Breast Cancer Wisconsin untuk klasifikasi."
    return df,target,desc

DATASETS=["Diabetes (Regression)","Iris (Classification / Clustering)","Wine (Classification)","Breast Cancer (Classification)"]
def xy(df,target):
    X=df.drop(columns=[target]).select_dtypes(include=np.number)
    return X,df[target]
def mongo_collection(collection="records"):
    if MongoClient is None: return None,"Paket pymongo belum terpasang."
    uri=""
    try: uri=st.secrets.get("MONGODB_URI","")
    except Exception: pass
    uri=uri or os.getenv("MONGODB_URI","")
    if not uri: return None,"MongoDB belum dikonfigurasi; aplikasi berjalan dalam mode lokal."
    try:
        client=MongoClient(uri,serverSelectionTimeoutMS=3000); client.admin.command("ping")
        db=client[os.getenv("MONGODB_DB","datasphere")]
        return db[collection],"Terhubung ke MongoDB Atlas."
    except Exception as e: return None,f"Koneksi MongoDB gagal: {e}"

with st.sidebar:
    st.markdown("## 🌐 DataSphere")
    st.caption("Explore Data. Discover Insights.")
    page=st.radio("Navigasi",["Home","Dataset Explorer","Machine Learning Lab","MongoDB & Retraining","Community","About & Methodology"])
    st.divider(); st.caption("AI Streamlit + MongoDB")

if page=="Home":
    st.markdown('<div class="hero"><h1>Welcome to DataSphere</h1><h3>Explore Data. Discover Insights.</h3><p>Platform data science interaktif untuk mengeksplorasi dataset, bereksperimen dengan Machine Learning, mengevaluasi model, dan berbagi wawasan.</p></div>',unsafe_allow_html=True)
    st.write(""); a,b,c=st.columns(3)
    with a: st.markdown("### 📊 Dataset Explorer"); st.write("Jelajahi fitur, distribusi, korelasi, dan kualitas data.")
    with b: st.markdown("### 🤖 Machine Learning Lab"); st.write("Latih model regresi, klasifikasi, dan clustering.")
    with c: st.markdown("### 🗄️ MongoDB & Retraining"); st.write("Simpan data tambahan dan uji ulang model.")
    d,e=st.columns(2)
    with d: st.markdown("### 💬 Community"); st.write("Bagikan komentar dan masukan.")
    with e: st.markdown("### 📘 About & Methodology"); st.write("Pelajari pendekatan dan arsitektur aplikasi.")
    st.info("Pilih halaman melalui menu di sebelah kiri.")

elif page=="Dataset Explorer":
    st.title("📊 Dataset Explorer"); st.write("Eksplorasi dataset sebelum membangun model.")
    name=st.selectbox("Pilih Dataset",DATASETS); df,target,desc=dataset(name); X,y=xy(df,target)
    c1,c2,c3,c4=st.columns(4); c1.metric("Samples",len(df)); c2.metric("Features",X.shape[1]); c3.metric("Missing Values",int(df.isna().sum().sum())); c4.metric("Data Type","Numeric")
    st.info(desc); t1,t2,t3,t4,t5=st.tabs(["Overview","Sample Data","Feature Distributions","Correlation","Data Quality"])
    with t1:
        l,r=st.columns(2)
        with l: st.plotly_chart(px.histogram(df,x=target,title="Distribusi Target"),use_container_width=True)
        with r: st.plotly_chart(px.imshow(X.corr(),aspect="auto",color_continuous_scale="RdBu_r",title="Korelasi Fitur"),use_container_width=True)
    with t2: st.dataframe(df.head(20),use_container_width=True)
    with t3:
        feature=st.selectbox("Pilih fitur",X.columns.tolist()); st.plotly_chart(px.histogram(df,x=feature,marginal="box"),use_container_width=True)
    with t4: st.dataframe(X.corr().round(3),use_container_width=True)
    with t5:
        st.dataframe(pd.DataFrame({"Tipe Data":df.dtypes.astype(str),"Missing":df.isna().sum(),"Nilai Unik":df.nunique()}),use_container_width=True)
        st.write("Duplikasi baris:",int(df.duplicated().sum()))
    st.download_button("⬇️ Unduh CSV",df.to_csv(index=False).encode(),"datasphere_dataset.csv","text/csv")

elif page=="Machine Learning Lab":
    st.title("🤖 Machine Learning Lab"); st.write("Latih dan evaluasi model dengan visualisasi interaktif.")
    tr,tc,tk=st.tabs(["Regression","Classification","Clustering"])
    with tr:
        st.subheader("Regression")
        model_name=st.selectbox("Model regresi",["Linear Regression","Random Forest Regressor"]); test=st.slider("Test Size",.1,.4,.2,.05,key="rt")
        if st.button("Train Regression Model",type="primary"):
            df,target,_=dataset("Diabetes (Regression)"); X,y=xy(df,target); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test,random_state=42)
            model=LinearRegression() if model_name=="Linear Regression" else RandomForestRegressor(n_estimators=150,random_state=42)
            model.fit(Xtr,ytr); pred=model.predict(Xte); mse=mean_squared_error(yte,pred)
            a,b,c,d=st.columns(4); a.metric("MAE",f"{mean_absolute_error(yte,pred):.2f}"); b.metric("MSE",f"{mse:,.2f}"); c.metric("RMSE",f"{np.sqrt(mse):.2f}"); d.metric("R²",f"{r2_score(yte,pred):.3f}")
            plot=pd.DataFrame({"Actual":yte,"Predicted":pred}); st.plotly_chart(px.scatter(plot,x="Actual",y="Predicted",title="Actual vs Predicted"),use_container_width=True); st.dataframe(plot.head(20))
    with tc:
        st.subheader("Classification"); name=st.selectbox("Dataset klasifikasi",DATASETS[1:],key="clfd"); model_name=st.selectbox("Model klasifikasi",["Logistic Regression","Random Forest Classifier"]); test=st.slider("Test Size",.1,.4,.2,.05,key="ct")
        if st.button("Train Classification Model",type="primary"):
            df,target,_=dataset(name); X,y=xy(df,target); strat=y if y.value_counts().min()>=2 else None
            Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test,random_state=42,stratify=strat)
            model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)) if model_name=="Logistic Regression" else RandomForestClassifier(n_estimators=150,random_state=42)
            model.fit(Xtr,ytr); pred=model.predict(Xte)
            a,b,c,d=st.columns(4); a.metric("Accuracy",f"{accuracy_score(yte,pred):.3f}"); b.metric("Precision macro",f"{precision_score(yte,pred,average='macro',zero_division=0):.3f}"); c.metric("Recall macro",f"{recall_score(yte,pred,average='macro',zero_division=0):.3f}"); d.metric("F1 macro",f"{f1_score(yte,pred,average='macro',zero_division=0):.3f}")
            labels=sorted(pd.Series(y).unique()); cm=confusion_matrix(yte,pred,labels=labels); st.plotly_chart(px.imshow(cm,text_auto=True,x=[str(i) for i in labels],y=[str(i) for i in labels],labels={"x":"Predicted","y":"Actual","color":"Count"},title="Confusion Matrix"),use_container_width=True)
    with tk:
        st.subheader("Clustering (K-Means)"); name=st.selectbox("Dataset clustering",DATASETS[1:3],key="kds"); k=st.slider("Jumlah cluster (K)",2,8,3)
        if st.button("Run K-Means",type="primary"):
            df,target,_=dataset(name); X,_=xy(df,target); scaled=StandardScaler().fit_transform(X); model=KMeans(n_clusters=k,random_state=42,n_init=10); labels=model.fit_predict(scaled)
            score=silhouette_score(scaled,labels); st.metric("Silhouette Score",f"{score:.3f}"); coords=PCA(n_components=2).fit_transform(scaled)
            plot=pd.DataFrame({"Component 1":coords[:,0],"Component 2":coords[:,1],"Cluster":labels.astype(str)}); st.plotly_chart(px.scatter(plot,x="Component 1",y="Component 2",color="Cluster",title="K-Means Cluster View"),use_container_width=True)

elif page=="MongoDB & Retraining":
    st.title("🗄️ MongoDB & Retraining"); col,status=mongo_collection(); st.success(status) if col is not None else st.warning(status)
    ta,tv,tr=st.tabs(["Add New Data","View Stored Data","Retrain Model"])
    with ta:
        with st.form("record"):
            age=st.number_input("Age",1,120,50); bmi=st.number_input("BMI",5.,80.,27.5); bp=st.number_input("Blood Pressure",40.,250.,80.); glucose=st.number_input("Glucose",20.,500.,120.); insulin=st.number_input("Insulin",0.,2000.,85.); skin=st.number_input("Skin Thickness",0.,100.,20.); submit=st.form_submit_button("Save to MongoDB",type="primary")
        if submit:
            rec={"age":age,"bmi":bmi,"blood_pressure":bp,"glucose":glucose,"insulin":insulin,"skin_thickness":skin,"created_at":datetime.utcnow().isoformat()+"Z"}
            if col is not None:
                try: col.insert_one(rec); st.success("Record berhasil disimpan ke MongoDB.")
                except Exception as e: st.error(str(e))
            else: st.session_state.setdefault("local_records",[]).insert(0,rec); st.success("Tersimpan sementara di sesi lokal.")
    with tv:
        records=[]
        if col is not None:
            try: records=list(col.find({}, {"_id":0}).sort("created_at",-1).limit(200))
            except Exception as e: st.error(str(e))
        else: records=st.session_state.get("local_records",[])
        st.dataframe(pd.DataFrame(records),use_container_width=True) if records else st.info("Belum ada record.")
    with tr:
        st.write("Demo melatih ulang baseline pada dataset Diabetes bawaan. Record formulir belum otomatis menjadi data latih karena skema fiturnya berbeda.")
        if st.button("Retrain baseline model",type="primary"):
            df,target,_=dataset("Diabetes (Regression)"); X,y=xy(df,target); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42); model=RandomForestRegressor(n_estimators=150,random_state=42); model.fit(Xtr,ytr); pred=model.predict(Xte); st.metric("Baseline RMSE",f"{np.sqrt(mean_squared_error(yte,pred)):.2f}")

elif page=="Community":
    st.title("💬 Community Hub"); st.write("Bagikan komentar dan masukan."); col,status=mongo_collection("comments")
    with st.form("comment_form",clear_on_submit=True):
        name=st.text_input("Your Name",placeholder="Anonymous Visitor"); comment=st.text_area("Your Comment",max_chars=300); post=st.form_submit_button("Post Comment",type="primary")
    if post:
        if not comment.strip(): st.error("Komentar tidak boleh kosong.")
        else:
            item={"name":name.strip() or "Anonymous Visitor","comment":comment.strip(),"created_at":datetime.utcnow().isoformat()+"Z"}
            if col is not None:
                try: col.insert_one(item); st.success("Komentar tersimpan di MongoDB.")
                except Exception as e: st.error(str(e))
            else: st.session_state.setdefault("local_comments",[]).insert(0,item); st.success("Komentar tersimpan sementara di sesi lokal.")
    comments=[]
    if col is not None:
        try: comments=list(col.find({}, {"_id":0}).sort("created_at",-1).limit(100))
        except Exception: pass
    else: comments=st.session_state.get("local_comments",[])
    st.subheader("Latest Discussions")
    for item in comments:
        with st.container(border=True): st.markdown(f"**{item.get('name','Anonymous Visitor')}** · {item.get('created_at','')}"); st.write(item.get("comment",""))
    if not comments: st.info("Belum ada komentar. Jadilah orang pertama yang berbagi.")

else:
    st.title("📘 About & Methodology"); ta,tm,ts,th=st.tabs(["About the Project","Methodology","System Architecture","Acknowledgement"])
    with ta:
        st.write("DataSphere adalah prototipe platform data science interaktif untuk eksplorasi dataset, pembelajaran Machine Learning, dan komentar komunitas.")
        st.table(pd.DataFrame({"Field":["Course","Technology","Year"],"Information":["Artificial Intelligence","Python, Streamlit, scikit-learn, Plotly, MongoDB","2026"]}))
    with tm:
        st.markdown("- **Exploration:** distribusi, korelasi, missing values, duplikasi.\n- **Regression:** MAE, MSE, RMSE, R².\n- **Classification:** accuracy, precision, recall, F1, confusion matrix.\n- **Clustering:** K-Means, standardisasi, silhouette score, PCA.\n- **Database:** MongoDB Atlas jika URI dikonfigurasi.\n- **Sentiment:** belum diimplementasikan sebagai model NLP.")
    with ts:
        st.code("Browser → Streamlit UI (app.py) → pandas / Plotly / scikit-learn\n                         └──────────────→ MongoDB Atlas (optional)",language="text")
        st.warning("Tanpa URI MongoDB, data hanya disimpan sementara di session aplikasi.")
    with th: st.write("Prototipe akademik. Sesuaikan identitas, informasi kampus, dan dokumentasi hasil pengujian sebelum presentasi.")

st.divider(); st.caption("DataSphere prototype • Python + Streamlit • Untuk tujuan pembelajaran")
