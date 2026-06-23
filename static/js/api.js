const API = {
    async getConfig() {
        const res = await fetch('/api/config');
        return res.json();
    },

    async getPumps() {
        const res = await fetch('/api/pumps');
        return res.json();
    },

    async calculate(data) {
        const res = await fetch('/api/calculate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Erro no cálculo');
        }
        return res.json();
    },
};
