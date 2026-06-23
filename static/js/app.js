let config = null;
let pumps = [];
let k2Factors = [];
let selectedPumps = new Set();

document.addEventListener('DOMContentLoaded', init);

async function init() {
    try {
        [config, pumps] = await Promise.all([API.getConfig(), API.getPumps()]);
        k2Factors = [...config.k2_defaults];
        renderMaterialSelect();
        renderK2Table();
        renderPumpSelector();
        document.getElementById('btn-calculate').addEventListener('click', runCalculation);
    } catch (e) {
        showError('Erro ao carregar configuração: ' + e.message);
    }
}

function renderMaterialSelect() {
    const sel = document.getElementById('material');
    sel.innerHTML = '';
    config.materials.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = m;
        if (m === 'PVC') opt.selected = true;
        sel.appendChild(opt);
    });
}

function renderK2Table() {
    const tbody = document.getElementById('k2-body');
    tbody.innerHTML = '';
    k2Factors.forEach((val, i) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${String(i).padStart(2, '0')}:00</td>
            <td><input type="number" step="0.01" min="0" value="${val}" data-idx="${i}" class="k2-input"></td>`;
        tbody.appendChild(tr);
    });
    tbody.querySelectorAll('.k2-input').forEach(inp => {
        inp.addEventListener('change', e => {
            k2Factors[parseInt(e.target.dataset.idx)] = parseFloat(e.target.value) || 0;
        });
    });
}

function renderPumpSelector() {
    const container = document.getElementById('pump-list');
    container.innerHTML = '';
    pumps.forEach(p => {
        const label = document.createElement('label');
        label.className = 'pump-checkbox';
        label.innerHTML = `<input type="checkbox" value="${p.sheet_name}" data-name="${p.name}">
            <span class="pump-name">${p.name}</span>`;
        label.querySelector('input').addEventListener('change', e => {
            if (e.target.checked) selectedPumps.add(p.sheet_name);
            else selectedPumps.delete(p.sheet_name);
        });
        container.appendChild(label);
    });
}

function getInputs() {
    const v = id => parseFloat(document.getElementById(id).value) || 0;
    return {
        consumo: v('consumo'),
        k1: v('k1'),
        k2_factors: k2Factors,
        l_suc: v('l_suc'),
        l_rec: v('l_rec'),
        sing_suc: v('sing_suc'),
        sing_rec: v('sing_rec'),
        h_suc: v('h_suc'),
        h_rec: v('h_rec'),
        d_suc: v('d_suc'),
        d_rec: v('d_rec'),
        t_fun: v('t_fun'),
        material: document.getElementById('material').value,
        tarifa: v('tarifa'),
        selected_pumps: [...selectedPumps],
    };
}

async function runCalculation() {
    const btn = document.getElementById('btn-calculate');
    const resultsEl = document.getElementById('results');
    btn.disabled = true;
    btn.textContent = 'Calculando...';
    resultsEl.classList.remove('visible');

    try {
        const inputs = getInputs();
        if (inputs.consumo <= 0 || inputs.k1 <= 0) {
            showError('Consumo e K1 devem ser maiores que zero.');
            return;
        }
        if (inputs.d_suc <= 0 || inputs.d_rec <= 0) {
            showError('Diâmetros devem ser maiores que zero.');
            return;
        }
        if (inputs.t_fun <= 0 || inputs.t_fun > 24) {
            showError('Tempo de funcionamento deve estar entre 0 e 24h.');
            return;
        }

        const result = await API.calculate(inputs);
        renderResults(result);
        resultsEl.classList.add('visible');
        resultsEl.scrollIntoView({ behavior: 'smooth' });
    } catch (e) {
        showError(e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Calcular';
    }
}

function renderResults(r) {
    // Demand chart
    Charts.renderDemand('chart-demand', r.demand_profile);

    // Diameter cards
    setMetric('metric-coef', r.diameters.coef_funcionamento.toFixed(4));
    setMetric('metric-decon', `${r.diameters.d_economico.toFixed(1)} mm`);
    setMetric('metric-dsuc-delta', formatDelta(r.diameters.delta_suc));
    setMetric('metric-drec-delta', formatDelta(r.diameters.delta_rec));

    // Velocity cards
    renderVelocity('vel-suc', 'Sucção', r.velocity_suc);
    renderVelocity('vel-rec', 'Recalque', r.velocity_rec);

    // Head losses
    setMetric('loss-lin-suc', `${r.head_losses.linear_suc.toFixed(4)} m`);
    setMetric('loss-lin-rec', `${r.head_losses.linear_rec.toFixed(4)} m`);
    setMetric('loss-sing-suc', `${r.head_losses.singular_suc.toFixed(4)} m`);
    setMetric('loss-sing-rec', `${r.head_losses.singular_rec.toFixed(4)} m`);
    setMetric('loss-total', `${r.head_losses.total.toFixed(4)} m`);
    setMetric('dh-geo', `${r.dh_geometrico.toFixed(2)} m`);
    setMetric('q-projeto', `${r.q_projeto.toFixed(2)} m³/h`);
    setMetric('h-projeto', `${r.h_projeto.toFixed(2)} m`);

    // System + pump chart
    if (r.pump_fits.length > 0) {
        document.getElementById('section-pumps-chart').style.display = 'block';
        Charts.renderSystemAndPumps('chart-system', r);
    } else {
        document.getElementById('section-pumps-chart').style.display = 'none';
    }

    // Intersections
    const interDiv = document.getElementById('intersections');
    interDiv.innerHTML = '';
    r.intersections.forEach(inter => {
        const cls = inter.above_optimal ? 'badge-success' : 'badge-warning';
        const icon = inter.above_optimal ? '✓' : '✗';
        const txt = inter.above_optimal ? 'acima' : 'abaixo';
        interDiv.innerHTML += `<div class="intersection-item">
            <strong>${inter.pump_name}</strong>: Q = ${inter.vazao} m³/h, H = ${inter.altura} m
            <span class="badge ${cls}">${icon} ${txt} do ponto ótimo</span>
        </div>`;
    });

    // Efficiency chart
    const hasEff = r.pump_fits.some(f => f.original_rendimento.length > 0);
    if (hasEff) {
        document.getElementById('section-efficiency').style.display = 'block';
        Charts.renderEfficiency('chart-efficiency', r);
    } else {
        document.getElementById('section-efficiency').style.display = 'none';
    }

    // Cost table
    const costBody = document.getElementById('cost-body');
    costBody.innerHTML = '';
    const sorted = [...r.efficiencies].sort((a, b) => a.custo_diario - b.custo_diario);
    sorted.forEach((eff, i) => {
        const tr = document.createElement('tr');
        if (i === 0) tr.className = 'best-row';
        tr.innerHTML = `<td>${i === 0 ? '🏆 ' : ''}${eff.pump_name}</td>
            <td>${eff.rendimento.toFixed(1)}%</td>
            <td>${eff.operating_hours.toFixed(1)}h</td>
            <td>R$ ${eff.custo_diario.toFixed(2)}</td>`;
        costBody.appendChild(tr);
    });
    document.getElementById('section-costs').style.display = sorted.length > 0 ? 'block' : 'none';
}

function setMetric(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function formatDelta(val) {
    const pct = (val * 100).toFixed(1);
    return val >= 0 ? `+${pct}%` : `${pct}%`;
}

function renderVelocity(id, label, vel) {
    const el = document.getElementById(id);
    if (!el) return;
    const statusClass = vel.status === 'ok' ? 'status-ok' : vel.status === 'high' ? 'status-high' : 'status-low';
    const statusLabel = vel.status === 'ok' ? 'Adequada' : vel.status === 'high' ? 'Acima do limite' : 'Abaixo do limite';
    el.innerHTML = `<div class="vel-value">${vel.value.toFixed(3)} m/s</div>
        <div class="vel-status ${statusClass}">${statusLabel}</div>`;
}

function showError(msg) {
    const el = document.getElementById('error-msg');
    el.textContent = msg;
    el.style.display = 'block';
    setTimeout(() => { el.style.display = 'none'; }, 5000);
}
