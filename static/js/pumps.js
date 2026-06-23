document.addEventListener('DOMContentLoaded', initPumps);

function initPumps() {
    // Tab navigation
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(btn.dataset.tab).classList.add('active');
        });
    });

    document.getElementById('btn-add-point').addEventListener('click', addCurvePoint);
    document.getElementById('btn-save-pump').addEventListener('click', savePump);
    document.getElementById('btn-preview-curve').addEventListener('click', previewCurve);

    // Start with 3 empty points
    for (let i = 0; i < 3; i++) addCurvePoint();

    loadCustomPumps();
}

function addCurvePoint() {
    const container = document.getElementById('curve-points');
    const idx = container.children.length + 1;
    const row = document.createElement('div');
    row.className = 'curve-point-row';
    row.innerHTML = `<span class="point-num">${idx}</span>
        <input type="number" class="cp-vazao" step="0.1" min="0" placeholder="0">
        <input type="number" class="cp-altura" step="0.1" min="0" placeholder="0">
        <input type="number" class="cp-rend" step="0.1" min="0" max="100" placeholder="">
        <button class="btn-remove-point" title="Remover">&times;</button>`;
    row.querySelector('.btn-remove-point').addEventListener('click', () => {
        row.remove();
        renumberPoints();
    });
    container.appendChild(row);
}

function renumberPoints() {
    document.querySelectorAll('#curve-points .point-num').forEach((el, i) => {
        el.textContent = i + 1;
    });
}

function getCurveData() {
    const rows = document.querySelectorAll('#curve-points .curve-point-row');
    const vazao = [], altura = [], rendimento = [];
    rows.forEach(row => {
        const v = parseFloat(row.querySelector('.cp-vazao').value);
        const a = parseFloat(row.querySelector('.cp-altura').value);
        const r = parseFloat(row.querySelector('.cp-rend').value);
        if (!isNaN(v) && !isNaN(a)) {
            vazao.push(v);
            altura.push(a);
            if (!isNaN(r)) rendimento.push(r);
        }
    });
    return { vazao, altura, rendimento };
}

function previewCurve() {
    const { vazao, altura, rendimento } = getCurveData();
    if (vazao.length < 2) {
        showError('Preencha pelo menos 2 pontos com vazao e altura.');
        return;
    }
    const chartEl = document.getElementById('chart-preview');
    chartEl.style.display = 'block';

    const traces = [
        {
            x: vazao, y: altura, mode: 'lines+markers', name: 'H (m)',
            line: { color: '#2e86de', width: 2 }, marker: { size: 8 },
        },
    ];
    if (rendimento.length === vazao.length) {
        traces.push({
            x: vazao, y: rendimento, mode: 'lines+markers', name: 'Rend. (%)',
            yaxis: 'y2', line: { color: '#27ae60', width: 2, dash: 'dot' }, marker: { size: 6 },
        });
    }

    Plotly.newPlot(chartEl, traces, {
        title: 'Curva da Bomba',
        xaxis: { title: 'Vazao (m3/h)' },
        yaxis: { title: 'Altura (m)' },
        yaxis2: rendimento.length === vazao.length ? {
            title: 'Rendimento (%)', overlaying: 'y', side: 'right',
        } : undefined,
        margin: { t: 40, r: 60, b: 50, l: 60 },
        showlegend: true,
        legend: { x: 0.5, y: -0.2, orientation: 'h', xanchor: 'center' },
    }, { responsive: true });
}

async function savePump() {
    const name = document.getElementById('pump-name').value.trim();
    if (!name) {
        showError('Informe o nome da bomba.');
        return;
    }
    const { vazao, altura, rendimento } = getCurveData();
    if (vazao.length < 3) {
        showError('Preencha pelo menos 3 pontos com vazao e altura.');
        return;
    }
    if (vazao.length !== altura.length) {
        showError('Vazao e altura devem ter o mesmo numero de pontos.');
        return;
    }

    const btn = document.getElementById('btn-save-pump');
    btn.disabled = true;
    btn.textContent = 'Salvando...';

    try {
        await API.createCustomPump({
            name,
            vazao,
            altura,
            rendimento: rendimento.length === vazao.length ? rendimento : [],
        });
        document.getElementById('pump-name').value = '';
        document.getElementById('curve-points').innerHTML = '';
        document.getElementById('chart-preview').style.display = 'none';
        for (let i = 0; i < 3; i++) addCurvePoint();
        await loadCustomPumps();
        await refreshPumpSelector();
        showSuccess('Bomba salva com sucesso!');
    } catch (e) {
        showError(e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Salvar Bomba';
    }
}

async function loadCustomPumps() {
    const container = document.getElementById('custom-pumps-list');
    try {
        const pumps = await API.getCustomPumps();
        if (pumps.length === 0) {
            container.innerHTML = '<p class="text-muted">Nenhuma bomba cadastrada ainda.</p>';
            return;
        }
        container.innerHTML = '';
        const table = document.createElement('table');
        table.className = 'cost-table';
        table.innerHTML = `<thead><tr>
            <th>Nome</th><th>Pontos</th><th>Rendimento</th><th></th>
        </tr></thead>`;
        const tbody = document.createElement('tbody');
        pumps.forEach(p => {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td>${p.name}</td>
                <td>${p.vazao.length} pontos</td>
                <td>${p.rendimento && p.rendimento.length > 0 ? 'Sim' : 'Nao'}</td>
                <td><button class="btn-delete" data-id="${p.id}">Excluir</button></td>`;
            tr.querySelector('.btn-delete').addEventListener('click', async () => {
                if (!confirm(`Excluir a bomba "${p.name}"?`)) return;
                try {
                    await API.deleteCustomPump(p.id);
                    await loadCustomPumps();
                    await refreshPumpSelector();
                } catch (e) {
                    showError(e.message);
                }
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        container.appendChild(table);
    } catch (e) {
        container.innerHTML = '<p class="text-muted">Erro ao carregar bombas. Banco de dados nao configurado.</p>';
    }
}

async function refreshPumpSelector() {
    try {
        pumps = await API.getPumps();
        renderPumpSelector();
    } catch (e) { /* ignore */ }
}

function showSuccess(msg) {
    const el = document.getElementById('error-msg');
    el.textContent = msg;
    el.style.display = 'block';
    el.style.background = '#e8f8f0';
    el.style.color = '#27ae60';
    el.style.borderLeftColor = '#27ae60';
    setTimeout(() => {
        el.style.display = 'none';
        el.style.background = '';
        el.style.color = '';
        el.style.borderLeftColor = '';
    }, 3000);
}
