const CHART_COLORS = {
    primary: '#1a2744',
    accent: '#2e86de',
    success: '#27ae60',
    warning: '#f39c12',
    danger: '#e74c3c',
    grid: '#e8edf2',
    text: '#2c3e50',
    traces: ['#2e86de', '#e74c3c', '#27ae60', '#f39c12', '#8e44ad', '#1abc9c', '#e67e22', '#3498db'],
};

const CHART_LAYOUT_BASE = {
    font: { family: 'Inter, system-ui, sans-serif', color: CHART_COLORS.text },
    paper_bgcolor: 'transparent',
    plot_bgcolor: '#fafbfc',
    margin: { l: 60, r: 30, t: 50, b: 50 },
    xaxis: { gridcolor: CHART_COLORS.grid, zerolinecolor: CHART_COLORS.grid },
    yaxis: { gridcolor: CHART_COLORS.grid, zerolinecolor: CHART_COLORS.grid },
    legend: { orientation: 'h', y: -0.15 },
};

const Charts = {
    renderDemand(containerId, demandProfile) {
        const horas = demandProfile.map(d => d.hora);
        const demandas = demandProfile.map(d => d.demanda);
        const trace = {
            x: horas,
            y: demandas,
            type: 'bar',
            marker: { color: CHART_COLORS.accent, borderRadius: 4 },
            name: 'Demanda (m³/h)',
        };
        const layout = {
            ...CHART_LAYOUT_BASE,
            title: { text: 'Perfil de Demanda Horária', font: { size: 16 } },
            xaxis: { ...CHART_LAYOUT_BASE.xaxis, title: 'Hora', dtick: 1 },
            yaxis: { ...CHART_LAYOUT_BASE.yaxis, title: 'Demanda (m³/h)' },
        };
        Plotly.newPlot(containerId, [trace], layout, { responsive: true });
    },

    renderSystemAndPumps(containerId, result) {
        const traces = [];

        // System curve
        traces.push({
            x: result.system_curve.map(p => p.vazao),
            y: result.system_curve.map(p => p.perda_carga),
            mode: 'lines',
            name: 'Curva do Sistema',
            line: { color: CHART_COLORS.primary, width: 3 },
        });

        // Pump curves
        result.pump_fits.forEach((fit, i) => {
            const color = CHART_COLORS.traces[i % CHART_COLORS.traces.length];
            traces.push({
                x: fit.original_vazao,
                y: fit.original_altura,
                mode: 'markers',
                name: `${fit.pump_name} (dados)`,
                marker: { color, size: 8, symbol: 'circle' },
            });
            traces.push({
                x: fit.fit_vazao,
                y: fit.fit_altura,
                mode: 'lines',
                name: `${fit.pump_name} (ajuste)`,
                line: { color, dash: 'dot', width: 2 },
            });
        });

        // Optimal point
        traces.push({
            x: [result.q_projeto],
            y: [result.h_projeto],
            mode: 'markers+text',
            name: 'Ponto Ótimo',
            marker: { color: CHART_COLORS.success, size: 14, symbol: 'diamond' },
            text: ['Ponto Ótimo'],
            textposition: 'top right',
            textfont: { color: CHART_COLORS.success, size: 12 },
        });

        // Intersection points
        result.intersections.forEach((inter, i) => {
            const color = CHART_COLORS.traces[i % CHART_COLORS.traces.length];
            traces.push({
                x: [inter.vazao],
                y: [inter.altura],
                mode: 'markers',
                name: `Interseção ${inter.pump_name}`,
                marker: { color, size: 10, symbol: 'x', line: { width: 2 } },
                showlegend: false,
            });
        });

        const layout = {
            ...CHART_LAYOUT_BASE,
            title: { text: 'Curva do Sistema × Bombas', font: { size: 16 } },
            xaxis: { ...CHART_LAYOUT_BASE.xaxis, title: 'Vazão (m³/h)', range: [0, result.q_projeto * 4] },
            yaxis: { ...CHART_LAYOUT_BASE.yaxis, title: 'Altura Manométrica (m)' },
            shapes: [
                { type: 'line', x0: result.q_projeto, x1: result.q_projeto, y0: 0, y1: result.h_projeto, line: { dash: 'dash', color: CHART_COLORS.success, width: 1.5 } },
                { type: 'line', x0: 0, x1: result.q_projeto, y0: result.h_projeto, y1: result.h_projeto, line: { dash: 'dash', color: CHART_COLORS.success, width: 1.5 } },
            ],
            annotations: [{
                x: result.q_projeto,
                y: 0,
                text: `Q = ${result.q_projeto} m³/h`,
                showarrow: false,
                yshift: -20,
                font: { size: 11, color: CHART_COLORS.success },
            }],
        };
        Plotly.newPlot(containerId, traces, layout, { responsive: true });
    },

    renderEfficiency(containerId, result) {
        const traces = [];
        result.pump_fits.forEach((fit, i) => {
            if (fit.original_rendimento.length === 0) return;
            const color = CHART_COLORS.traces[i % CHART_COLORS.traces.length];
            traces.push({
                x: fit.original_vazao.slice(0, fit.original_rendimento.length),
                y: fit.original_rendimento,
                mode: 'markers',
                name: `${fit.pump_name} (dados)`,
                marker: { color, size: 8 },
            });
            if (fit.fit_rendimento_vazao.length > 0) {
                traces.push({
                    x: fit.fit_rendimento_vazao,
                    y: fit.fit_rendimento_values,
                    mode: 'lines',
                    name: `${fit.pump_name} (ajuste)`,
                    line: { color, dash: 'dot', width: 2 },
                });
            }
        });

        // Mark operating points
        result.efficiencies.forEach((eff, i) => {
            const inter = result.intersections.find(x => x.pump_name === eff.pump_name);
            if (!inter) return;
            const color = CHART_COLORS.traces[i % CHART_COLORS.traces.length];
            traces.push({
                x: [inter.vazao],
                y: [eff.rendimento],
                mode: 'markers+text',
                name: `η ${eff.pump_name}`,
                marker: { color, size: 12, symbol: 'star' },
                text: [`${eff.rendimento.toFixed(1)}%`],
                textposition: 'top center',
                showlegend: false,
            });
        });

        const layout = {
            ...CHART_LAYOUT_BASE,
            title: { text: 'Curvas de Rendimento', font: { size: 16 } },
            xaxis: { ...CHART_LAYOUT_BASE.xaxis, title: 'Vazão (m³/h)' },
            yaxis: { ...CHART_LAYOUT_BASE.yaxis, title: 'Rendimento (%)' },
        };
        Plotly.newPlot(containerId, traces, layout, { responsive: true });
    },
};
