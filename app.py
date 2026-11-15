import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import io

st.set_page_config(
    page_title="Enterprise Real Estate AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background-color: #f8f9fc; }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #0d6efd;
    }
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        border-radius: 6px;
        padding: 12px 24px;
        font-weight: 600;
        width: 100%;
        border: none;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #0b5ed7;
        box-shadow: 0 4px 12px rgba(13, 110, 253, 0.2);
    }
    .header-container { display: flex; align-items: center; padding-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl", "rb"))
    columns = pickle.load(open("columns.pkl", "rb"))
    return model, columns

@st.cache_data
def load_data():
    return pd.read_csv("boston.csv")

model, columns = load_models()
historical_data = load_data()

col_menu, col_title = st.columns([1, 11])
with col_menu:
    st.write("")
    with st.popover("☰ Menu"):
        st.markdown("**System Architecture**")
        st.markdown("""
        * **Algorithm**: Linear Regression
        * **Preprocessing**: Standard & MinMax Scaling
        * **Explainability**: Coefficient Analysis
        * **Data**: Boston Housing Dataset
        """)
        st.divider()
        st.caption("v3.0 - Enterprise Edition")

with col_title:
    st.markdown('<h1 style="color: #1e293b; font-weight: 800; margin-bottom: 0;">🏢 Enterprise Property Valuation Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; font-size: 1.1rem;">AI-driven real estate analytics, market insights, and predictive pricing.</p>', unsafe_allow_html=True)

tab_valuation, tab_insights, tab_explain = st.tabs(["🏡 Valuation Engine", "📊 Market Insights", "🧠 AI Explainability"])

with tab_valuation:
    st.markdown("### Property Specifications")
    
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        rm = c1.number_input("Rooms/Dwelling (RM)", value=6.0, step=0.1)
        age = c2.number_input("Pre-1940 Units % (AGE)", value=65.2, step=1.0)
        lstat = c3.number_input("Lower Status % (LSTAT)", value=4.98, step=0.1)
        ptratio = c4.number_input("Pupil-Teacher Ratio", value=15.3, step=0.1)

    with st.expander("📍 Advanced Location & Zoning Parameters"):
        c_adv1, c_adv2, c_adv3 = st.columns(3)
        crim = c_adv1.number_input("Crime Rate (CRIM)", value=0.006, step=0.01, format="%.4f")
        zn = c_adv2.number_input("Zoned Land (ZN)", value=18.0, step=1.0)
        indus = c_adv3.number_input("Industrial Acres (INDUS)", value=2.31, step=0.1)
        
        nox = c_adv1.number_input("Nitric Oxides (NOX)", value=0.538, step=0.01, format="%.3f")
        dis = c_adv2.number_input("Distance to Employment (DIS)", value=4.09, step=0.1)
        rad = c_adv3.number_input("Highway Access Index (RAD)", value=1.0, step=1.0)
        
        tax = c_adv1.number_input("Property Tax Rate (TAX)", value=296.0, step=10.0)
        b = c_adv2.number_input("Demographic Factor (B)", value=396.9, step=1.0)
        chas = c_adv3.selectbox("River Bound (CHAS)", options=[0, 1])

    input_dict = {
        'CRIM': crim, 'ZN': zn, 'INDUS': indus, 'CHAS': chas, 'NOX': nox,
        'RM': rm, 'AGE': age, 'DIS': dis, 'RAD': rad, 'TAX': tax,
        'PTRATIO': ptratio, 'B': b, 'LSTAT': lstat
    }
    
    for col in columns:
        if col not in input_dict:
            input_dict[col] = 0.0

    st.markdown("<br>", unsafe_allow_html=True)
    c_btn1, c_btn2, c_btn3 = st.columns([1, 2, 1])
    
    with c_btn2:
        if st.button("Generate Valuation Report", use_container_width=True):
            df_input = pd.DataFrame([input_dict], columns=columns)
            try:
                raw_pred = model.predict(df_input)[0]
                
                # Assume raw prediction is in $1000s based on standard Boston dataset
                usd_value = raw_pred * 1000 
                # Convert to Indian Rupees (Assuming 1 USD = ₹83 for current standards)
                inr_value = (usd_value * 83) / 100000 
                
                st.markdown("---")
                st.markdown("### Valuation Results")
                
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.markdown(f'''
                    <div class="metric-card">
                        <p style="color: #64748b; margin: 0; font-size: 0.9rem; font-weight: 600;">ESTIMATED VALUE (USD)</p>
                        <h2 style="color: #0f172a; margin: 5px 0 0 0;">${usd_value:,.0f}</h2>
                    </div>
                    ''', unsafe_allow_html=True)
                with res_col2:
                    st.markdown(f'''
                    <div class="metric-card">
                        <p style="color: #64748b; margin: 0; font-size: 0.9rem; font-weight: 600;">ESTIMATED VALUE (INR)</p>
                        <h2 style="color: #0f172a; margin: 5px 0 0 0;">₹{inr_value:,.2f} Lakhs</h2>
                    </div>
                    ''', unsafe_allow_html=True)
                with res_col3:
                    st.markdown(f'''
                    <div class="metric-card">
                        <p style="color: #64748b; margin: 0; font-size: 0.9rem; font-weight: 600;">PRICE PER ROOM (USD)</p>
                        <h2 style="color: #0f172a; margin: 5px 0 0 0;">${(usd_value/rm):,.0f}</h2>
                    </div>
                    ''', unsafe_allow_html=True)

                csv_data = df_input.copy()
                csv_data['Predicted_USD'] = usd_value
                csv_data['Predicted_INR_Lakhs'] = inr_value
                csv_buffer = io.StringIO()
                csv_data.to_csv(csv_buffer, index=False)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="📥 Export Financial Report (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="valuation_report.csv",
                    mime="text/csv",
                )
                
            except Exception as e:
                st.error(f"Prediction Pipeline Failed: {e}")

with tab_insights:
    st.markdown("### Market Data & Trends")
    st.write("Explore historical relationships between property features and final pricing.")
    
    graph_col1, graph_col2 = st.columns(2)
    
    with graph_col1:
        fig1 = px.scatter(
            historical_data, x='RM', y='Price', 
            color='Price', color_continuous_scale='Viridis',
            title="Room Count vs. Market Price",
            labels={'RM': 'Avg Rooms', 'Price': 'Price ($1000s)'}
        )
        fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig1, use_container_width=True)
        
    with graph_col2:
        fig2 = px.scatter(
            historical_data, x='LSTAT', y='Price', 
            color='Price', color_continuous_scale='Inferno',
            title="Lower Income Status vs. Market Price",
            labels={'LSTAT': '% Lower Status Population', 'Price': 'Price ($1000s)'}
        )
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

with tab_explain:
    st.markdown("### AI Decision Drivers")
    st.write("Understand which parameters are positively or negatively impacting the valuation algorithm based on the trained linear coefficients.")
    
    try:
        coefficients = model.coef_
        feat_importance = pd.DataFrame({
            'Feature': columns,
            'Impact Coefficient': coefficients
        }).sort_values(by='Impact Coefficient', key=abs, ascending=True)
        
        fig_importance = px.bar(
            feat_importance, 
            x='Impact Coefficient', 
            y='Feature', 
            orientation='h',
            title="Global Feature Importance (Linear Regression)",
            color='Impact Coefficient',
            color_continuous_scale=['red', 'gray', 'green']
        )
        fig_importance.update_layout(plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_importance, use_container_width=True)
        
        st.info("💡 **Interpretation:** Green bars represent features that drive property prices UP (e.g., number of rooms). Red bars represent features that drive prices DOWN (e.g., higher nitric oxide pollution or higher crime rates).")
    except Exception as e:
        st.warning(f"Could not load feature importance chart: {e}")