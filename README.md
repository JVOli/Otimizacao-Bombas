## Otimização de Bombas para Reservatórios Elevados

Aplicação interativa (Streamlit) desenvolvida no contexto do TCC "Otimização de Bombas de água para reservatórios elevados" (Autor: João Vitor de Oliveira Cunha). O app auxilia no dimensionamento e na escolha de bombas, estimando perdas de carga, ponto de operação, rendimento e custo diário de energia com base em dados hidráulicos e curvas de bombas fornecidas via planilha Excel.

### Principais funcionalidades
- **Edição do perfil horário de demanda**: ajuste do fator K2 ao longo de 24 horas e do consumo base.
- **Cálculo hidráulico**: diâmetro econômico, velocidades, perdas de carga (lineares e singulares) e desnível geométrico.
- **Curva da rede**: traçado da curva da rede e identificação do ponto de operação desejado.
- **Curvas das bombas**: leitura de curvas a partir do Excel, ajuste polinomial (grau 2) e cálculo do ponto de interseção com a rede.
- **Rendimento e custo**: previsão de rendimento no ponto de operação e estimativa do custo diário de energia por bomba, com ordenação crescente.

### Requisitos
- Python 3.9+
- Bibliotecas em `requirements.txt`:
  - streamlit, pandas, numpy, plotly, openpyxl, scikit-learn, pybase64

Instale as dependências:
```bash
pip install -r requirements.txt
```

### Como executar
1. Garanta que os arquivos necessários estejam na raiz do projeto: `main.py`, `Bomba.xlsx`, `image.png`, `logotipo_iph.png` e a imagem de assets indicada no código.
2. Execute o app Streamlit:
```bash
streamlit run main.py
```
3. Acesse o endereço exibido no terminal (normalmente `http://localhost:8501`).

### Entradas e parâmetros (no app)
- **Demanda**: `Consumo base (m³/h)` e `Fator maior dia (k1)`; tabela editável de `Fator K2` por hora.
- **Rede (sidebar → Dados rede)**: comprimentos (`L_suc`, `L_rec`), singularidades (`Sing_suc`, `Sing_rec`), cotas (`H_suc`, `H_rec`), diâmetros (`D_suc`, `D_rec` em mm), tempo de funcionamento diário (`T_fun`), material da tubulação e tarifa de energia (`Tarifa`, em R$/kWh).
- **Diâmetros e velocidades**: o app calcula e mostra o diâmetro econômico, compara diâmetros informados e verifica faixas recomendadas de velocidade (≈0,6 a 3,0 m/s).

### Curvas de bombas (arquivo Bomba.xlsx)
O arquivo `Bomba.xlsx` deve conter:
- **Aba principal (catálogo)**: usada para listar as bombas e montar as opções de seleção. O app lê essa aba (primeira aba do arquivo) e cria a lista combinando as duas primeiras colunas (ex.: `Modelo` e `Potência`).
- **Uma aba por bomba**: nome da aba deve corresponder ao rótulo exibido no catálogo, com as seguintes normalizações aplicadas pelo app ao nome selecionado: substituição de `/` por `_` e remoção de `cv`. Ex.: `"Bomba X/10cv"` → aba `Bomba X_10`.
- **Colunas esperadas nas abas de curvas** (após a transposição usada pelo app):
  - `Vazão em m³/h válida para sucção de 0 m.c.a.`
  - `Altura Manométric Total (m.c.a.)`
  - `Rendimento %`

Observações importantes:
- Os dados devem ser numéricos, sem unidades no corpo da tabela (apenas no cabeçalho).
- O app ajusta uma regressão polinomial de grau 2 para estimar a curva da bomba e do rendimento, e busca a interseção com a curva da rede para obter o ponto de operação.
- O custo diário de energia é estimado a partir da vazão/altura no ponto de operação e da tarifa informada.

### Arquivos do projeto
- `main.py`: aplicação Streamlit principal.
- `Bomba.xlsx`: planilha com catálogo e curvas das bombas.
- `image.png`: imagem exibida no cabeçalho.
- `logotipo_iph.png`: logotipo exibido como marca d'água/rodapé.
- `assets_task_*.png`: imagem usada na sidebar.
- `requirements.txt`: dependências Python.
- `TCC - Versão Final João Vitor de Oliveira Cunha.pdf`: documento completo do trabalho.
- `LICENSE`: licença do repositório.

### Estrutura de cálculo (resumo)
- Coeficiente de funcionamento: `T_fun/24`.
- Diâmetro econômico: `1.3 * (T_fun/24)^(1/4) * ((Consumo*k1)/3600)^(1/2) * 1000`.
- Velocidades em sucção/recalque: `V = (Consumo*k1) / (3600 * (D/2000)^2 * π)`.
- Perdas lineares: fórmula de Hazen-Williams com `Coef_HW` conforme material selecionado.
- Perdas singulares: `ΔH = V^2 * K / (2*g)`.
- Curva da rede: `H = dH_geo + K * Q^1.852` (com `Q` em m³/h e constantes calculadas no app).
- Ponto de operação desejado: linha vertical em `Q = Consumo*k1*24/T_fun` com linha horizontal correspondente de altura.

### Dicas e solução de problemas
- Se aparecerem mensagens como "Erro ao plotar curva da bomba" ou "Nenhuma interseção encontrada", verifique:
  - Nomes das abas das bombas e formatação (ver normalizações acima).
  - Presença e nomes exatos das colunas exigidas.
  - Se as células contêm apenas números (sem textos/unidades misturados).
- Para ler o Excel é necessário `openpyxl` (já listado em `requirements.txt`).

### Licença
Este projeto é distribuído sob a licença informada em `LICENSE`.

### Créditos
Trabalho de Conclusão de Curso de João Vitor de Oliveira Cunha. Caso utilize este projeto em publicações, cite o autor e o TCC. 
