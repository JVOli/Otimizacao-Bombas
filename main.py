#Imports
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_icon="image.png",
    layout="wide",
)
#
col1, col2 = st.columns((1,3))
k2 = [0.74,0.66,0.61,0.59,0.58,0.60,0.77,0.94,1.08,1.20, 
      1.31,1.35,1.32,1.26,1.2,1.14,1.12,1.13,1.21,1.26,
      1.17,1.07,0.96,0.86]
df = pd.DataFrame(
    [
        {"Horário": i, "Fator K2": k2[i] } for i in range(0,24)
    ]
)
df.set_index('Horário', inplace=True)
with col1:
    Consumo = st.number_input("Consumo base m³/h")
    k1 = st.number_input("Fator Maior dia de Consumo")
    edited_df = st.data_editor(df)
with col2:
    st.line_chart(edited_df * k1 * Consumo)
    #st.markdown(list(float(edited_df.iloc[i]["Fator K2"]) for i in range(0,24)))
Coef_HW = {"Aço": 135, "Aço Galvanizado": 125, "Cobre": 130, "Chumbo": 130, "Latão": 130, "PVC": 140, "Ferro Fundido Revestido": 130, "Ferro Fundido Novo": 125, "Ferro Fundido Usado": 90, "Concreto": 120}
with st.sidebar.expander("Dados rede"):
    L_suc = st.number_input("Comprimento de sucção da rede em metros:")
    L_rec = st.number_input("Comprimento de recalque da rede em metros:")
    Sing_suc = st.number_input("Singularidades na sucção da rede")
    Sing_rec = st.number_input("Singularidades no recalque da rede")
    H_suc = st.number_input("Cota da sucção em metros:")
    H_rec = st.number_input("Cota do recalque em metros:")
    D_suc = st.number_input("Diâmetro da sucção em milímetros:")
    D_rec = st.number_input("Diâmetro do recalque em milímetros:")
    T_fun = st.number_input("Tempo de funcionamento diário desejado:")
    Mat = st.selectbox("Material da tubulação",["Aço", "Aço Galvanizado", "Cobre", "Chumbo", "Latão", "PVC", "Ferro Fundido Revestido", "Ferro Fundido Novo", "Ferro Fundido Usado", "Concreto"])
    Tarifa = st.number_input("Tarifa do kWh em R$:")
with st.sidebar.expander("Diâmetros (mm)"):
    Coef_fun = T_fun / 24
    st.metric("Coeficiente de funcionamento", Coef_fun)
    D_econ =1.3 * Coef_fun ** (1/4) * (Consumo * k1 /3600) ** 0.5 * 1000
    st.metric("Diâmetro economico", D_econ)
    st.metric("Sucção",D_suc,delta = (D_suc)/D_econ-1)
    st.metric("Recalque",D_rec,delta = (D_rec)/D_econ-1)
with st.sidebar.expander("Velociddes (m/s)"):
    V_op_suc = Consumo * k1 / (3600 * (D_suc/2000) ** 2 * 3.1415)
    V_op_rec = Consumo * k1 / (3600 * (D_rec/2000) ** 2 * 3.1415)
    if(V_op_suc > 3): 
        st.metric("Sucção",V_op_suc, delta=(V_op_suc)/3, delta_color= "inverse")
    elif(V_op_suc < 0.6):
        st.metric("Sucção",V_op_suc, delta=(V_op_suc)/0.6,)
    else:
        st.metric("Sucção",V_op_suc)
    if(V_op_rec > 3): 
        st.metric("Recalque",V_op_rec, delta=(V_op_rec)/3, delta_color= "inverse")
    elif(V_op_rec < 0.6):
        st.metric("Recalque",V_op_rec, delta=(V_op_rec)/0.6)
    else:
        st.metric("Recalque",V_op_rec)
col1, col2, col3, col4 = st.columns((1,1,1,1))
with col1:
    Coef_lin_suc = (10.643*((1/3600)**1.85))/((Coef_HW[Mat]**1.85)*((D_rec/1000)**4.87))
    st.metric("Perda de Carga Linear na Sucção", f'{round(Coef_lin_suc,5) * L_suc}')
with col2:
    Coef_lin_rec = (10.643*((1/3600)**1.85))/((Coef_HW[Mat]**1.85)*((D_rec/1000)**4.87))
    st.metric("Perda de Carga Linear no Recalque", f'{round(Coef_lin_rec,5) * L_rec}')
with col3:
    dH_suc = V_op_suc**2*Sing_suc/(2*9.81)
    st.metric("Perda de Carga Singular na Sucção", f'{round(dH_suc,3)} m')
with col4:
    dH_rec = V_op_rec**2*Sing_rec/(2*9.81)
    st.metric("Perda de Carga Singular no recalque", f'{round(dH_rec,3)} m')
col1, col2 = st.columns((1,1))
dH_Tot =Coef_lin_rec * L_rec + Coef_lin_suc *L_suc + dH_rec + dH_suc
with col1:
    st.metric("Perda de caerga toal", dH_Tot)
dH_geo = H_rec - H_suc
with col2:
    st.metric("Desnivel Geométrico", dH_geo)
CoefK = dH_Tot / (Consumo * k1) ** 1.852
Vazao = np.arange(0, (Consumo * k1 * 4*24/T_fun), 1)
dH_Vazao = dH_geo + CoefK * Vazao ** 1.852
df_vazao = pd.DataFrame(
    {
        "Vazão": Vazao,
        "Perda de Carga": dH_Vazao
    }
) 
# --- Combined Plotly Chart ---
fig = go.Figure()

Tabela_Bombas = pd.read_excel("Bomba.xlsx")
st.dataframe(Tabela_Bombas)
Bombas = list(str(Tabela_Bombas.iloc[i,0]) + ' - ' + str(Tabela_Bombas.iloc[i,1]) for i in range(0,np.shape(Tabela_Bombas)[0]))
#st.write(Bombas)
Selecao = st.multiselect("Escolha as bombas", Bombas)
Selecao = list(i.replace('/', '_').replace('cv', '') for i in Selecao)
#st.write(Selecao)
Curva = pd.read_excel("Bomba.xlsx", sheet_name=Selecao, index_col=None, header=None)
bomba_functions = []
Intersecao = {}
if Curva and len(Selecao) > 0:
    for Bomba in list(Curva.keys()):
        bomba_df = Curva[Bomba]
        try:
            #st.write(bomba_df.iloc[1])
            bomba_vazao = pd.DataFrame(bomba_df.iloc[0][1:]).values.reshape(-1, 1)
            bomba_altura = pd.DataFrame(bomba_df.iloc[1][1:]).values.reshape(-1, 1)
            #st.write(1)
            # Remove NaNs
            X = bomba_vazao
            y = bomba_altura
            # Fit polynomial regression (degree 2)
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            model = LinearRegression().fit(X_poly, y)
            # Store function
            f = lambda x, m=model, p=poly: m.predict(p.transform(np.array(x).reshape(-1, 1)))
            bomba_functions.append(f)
            st.info(f"Coeficientes da bomba {Bomba}: {model.coef_}, Intercepto: {model.intercept_}")
            # Plot fitted line on the main figure
            x_fit = np.linspace(float(X.min()), float(X.max()), 100)
            y_fit = f(x_fit)
            fig.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=y_fit.flatten(),
                    mode='lines',
                    name=f'Ajuste {Bomba}',
                    line=dict(dash='dot')
                )
            )
            # Find intersection between fitted line and df_vazao
            try:
                x_vazao = df_vazao['Vazão'].values
                y_rede = df_vazao['Perda de Carga'].values
                y_bomba_fit = f(x_vazao).flatten()
                diff = y_bomba_fit - y_rede
                sign_change = np.where(np.diff(np.sign(diff)))[0]
                if len(sign_change) == 0:
                    st.info(f"Nenhuma interseção (ajuste) encontrada para a bomba {Bomba}.")
                else:
                    for idx in sign_change:
                        x0, x1 = x_vazao[idx], x_vazao[idx+1]
                        y0, y1 = diff[idx], diff[idx+1]
                        if y1 - y0 != 0:
                            x_intersect = x0 - y0 * (x1 - x0) / (y1 - y0)
                            y_intersect = np.interp(x_intersect, x_vazao, y_rede)
                            st.success(f"Interseção Ajuste {Bomba}: Vazão = {x_intersect:.2f} m³/h, Altura/Perda de Carga = {y_intersect:.2f} m")
                            # Check if y_intersect > Consumo * k1
                            if x_intersect > Consumo * k1*24/T_fun:
                                st.info(f"✓ Interseção Ajuste {Bomba} está ACIMA do ponto ótimo ({Consumo * k1*24/T_fun:.2f})")
                            else:
                                st.warning(f"✗ Interseção Ajuste {Bomba} está ABAIXO do ponto ótimo ({Consumo * k1*24/T_fun:.2f})")
                            Intersecao[Bomba] = {"Vazão": x_intersect, "Perda de Carga": y_intersect}
            except Exception as e:
                st.warning(f"Erro ao calcular interseção do ajuste para bomba {Bomba}: {e}")
        except Exception as e:
            st.warning(f"Erro ao ajustar curva da bomba {Bomba}: {e}")
        Curva[Bomba] = Curva[Bomba].transpose().rename(columns=Curva[Bomba].iloc[:,0]).drop(Curva[Bomba].index[0])
        st.dataframe(Curva[Bomba])
        #st.write(Curva[Bomba])
        # Add bomba curve to plot
        try:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_numeric(Curva[Bomba]['Vazão em m³/h válida para sucção de 0 m.c.a.'], errors='coerce'),
                    y=pd.to_numeric(Curva[Bomba]['Altura Manométric Total (m.c.a.)'], errors='coerce'),
                    mode='markers',
                    name=f'Curva {Bomba}'
                )
            )
        except Exception as e:
            st.warning(f"Erro ao plotar curva da bomba {Bomba}: {e}")

# Add toggle for vertical line
show_vline = st.checkbox("Mostrar linha vertical em Consumo * k1", value=True)
# Add df_vazao curve
fig.add_trace(go.Scatter(x=df_vazao["Vazão"], y=df_vazao["Perda de Carga"], mode='lines', name='Perda de Carga (Rede)'))

fig.update_layout(
    title="Curva da Rede e Bombas",
    xaxis_title="Vazão (m³/h)",
    yaxis_title="Altura/Perda de Carga (m)",
    legend_title="Curvas",
    xaxis=dict(range=[0, 4*Consumo*k1*24/T_fun])
)
# Add vertical line at Consumo * k1 if toggle is on
if show_vline:
    fig.add_vline(
        x=Consumo * k1*24/T_fun,
        line_dash="dash",
        annotation_text=f"Consumo * k1*Consumo*T_fun/24 = {Consumo * k1*24/T_fun:.2f}",
        annotation_position="top right"
    )
    # Add horizontal line at Y = dH_geo + CoefK * (k1*Consumo) ** 1.852
    y_hline = dH_geo + CoefK * (k1*Consumo*24/T_fun) ** 1.852
    fig.add_hline(
        y=y_hline,
        line_dash="dash",
        annotation_text=f"Altura/Perda = {y_hline:.2f}",
        annotation_position="right"
    )
    # Add 'Ponto Otimo' marker
    fig.add_trace(
        go.Scatter(
            x=[Consumo * k1*24/T_fun],
            y=[y_hline],
            mode='markers+text',
            marker=dict(color='green', size=12, symbol='diamond'),
            name='Ponto Otimo',
            text=["Ponto Otimo"],
            textposition="top right"
        )
    )

st.plotly_chart(fig)
#st.dataframe(Curva[Bomba])

# Create rendimento plot for each bomba
Custo_Energia = {}
if Curva and len(Selecao) > 0:
    fig_rendimento = go.Figure()
    for Bomba in list(Curva.keys()):
        try:
            if (k1*Consumo*24) / Intersecao[Bomba]["Vazão"] <=24:
                bomba_df = Curva[Bomba]
                # Extract rendimentos
                vazoes = pd.DataFrame(bomba_df.iloc[:,0]).values.reshape(-1, 1)
                rendimentos = pd.DataFrame(bomba_df.iloc[:,2]).values.reshape(-1, 1)
                #st.write(vazoes)
                #st.write(rendimentos)
                fig_rendimento.add_trace(
                    go.Scatter(
                        x=pd.to_numeric(Curva[Bomba]['Vazão em m³/h válida para sucção de 0 m.c.a.'], errors='coerce'),
                        y=pd.to_numeric(Curva[Bomba]['Rendimento %'], errors='coerce'),
                        mode='markers',
                        name=f'Curva {Bomba}'
                    )
                )
                # Fit polynomial to rendimento curve
                try:
                    x_rend = pd.to_numeric(Curva[Bomba]['Vazão em m³/h válida para sucção de 0 m.c.a.'], errors='coerce').values.reshape(-1, 1)
                    y_rend = pd.to_numeric(Curva[Bomba]['Rendimento %'], errors='coerce').values
                    # Remove NaNs
                    mask = ~np.isnan(x_rend.flatten()) & ~np.isnan(y_rend)
                    X_rend = x_rend[mask].reshape(-1, 1)
                    y_rend_clean = y_rend[mask]
                    # Fit polynomial regression (degree 2)
                    poly_rend = PolynomialFeatures(degree=2)
                    X_poly_rend = poly_rend.fit_transform(X_rend)
                    model_rend = LinearRegression().fit(X_poly_rend, y_rend_clean)
                    # Plot fitted line
                    x_fit_rend = np.linspace(float(X_rend.min()), float(X_rend.max()), 100)
                    f_rend = lambda x, m=model_rend, p=poly_rend: m.predict(p.transform(np.array(x).reshape(-1, 1)))
                    y_fit_rend = f_rend(x_fit_rend)
                    fig_rendimento.add_trace(
                        go.Scatter(
                            x=x_fit_rend,
                            y=y_fit_rend.flatten(),
                            mode='lines',
                            name=f'Ajuste Rendimento {Bomba}',
                            line=dict(dash='dot')
                        )
                    )
                    # Predict rendimento for x=3
                    rendimento_pred = f_rend(Intersecao[Bomba]["Vazão"])
                    st.info(f"Rendimento previsto para Vazão={Intersecao[Bomba]["Vazão"]:.2f} m³/h da bomba {Bomba}: {rendimento_pred[0]:.2f}%")
                    Custo_Energia[f'{Bomba}'] = Intersecao[Bomba]["Vazão"] * Intersecao[Bomba]["Perda de Carga"] * 9800 * Tarifa * (k1*Consumo*24) / rendimento_pred / Intersecao[Bomba]["Vazão"] / 3600 /1000 * 100
                    #st.write(Intersecao[Bomba]["Vazão"], Intersecao[Bomba]["Perda de Carga"], Tarifa, (k1*Consumo*24), rendimento_pred)
                    st.write(f'Tempo de Operação Diário: {(k1*Consumo*24) / Intersecao[Bomba]["Vazão"]}')
                except Exception as e:
                    st.warning(f"Erro ao ajustar curva de rendimento da bomba {Bomba}: {e}")
        except Exception as e:
            st.warning(f"Erro ao extrair rendimentos da bomba {Bomba}: {e}")
    
    fig_rendimento.update_layout(
        title="Rendimentos das Bombas",
        xaxis_title="Vazão (m³/h)",
        yaxis_title="Rendimento (%)",
        legend_title="Bombas"
    )
    st.plotly_chart(fig_rendimento)

# Convert Custo_Energia to DataFrame
custo_df = pd.DataFrame(list(Custo_Energia.items()), columns=['Bomba', 'Custo diário'])
# Sort by 'Custo diário'
custo_df = custo_df.sort_values(by='Custo diário')
# Show in Streamlit
st.dataframe(custo_df, use_container_width=True)

st.markdown(
    f"""
    <style>
        .footer-logo {{
            position: fixed;
            right: 30px;
            bottom: 30px;
            z-index: 100;
        }}
    </style>
    <div class="footer-logo">
        <img src="data:image/png;base64,{logo_base64}" width="80">
    </div>
    """,
    unsafe_allow_html=True
)
