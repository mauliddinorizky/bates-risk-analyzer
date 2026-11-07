"""
Analisis Risiko Harga Saham - Model Bates (SV + Jumps), dengan Monte Carlo dan
Fungsi Densitas Transisi (TDF).
Jalankan:  streamlit run app.py
Dependensi: pip install streamlit yfinance numpy pandas scipy matplotlib
"""
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from scipy.integrate import quad_vec
from matplotlib.figure import Figure

TD = 252  # hari kerja per tahun


# ---------------------------------------------------------------- DATA
@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(ticker: str, period: str):
    import yfinance as yf
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.dropna().astype(float)


# ---------------------------------------------------------- ESTIMASI
def cir_ols(V, dt, scale_theta):
    """OLS CIR (skripsi Pers. 2.17-2.20): dV/sqrt(V) = a*dt/sqrt(V) + b*sqrt(V)*dt + e."""
    Vp, Vn = V[:-1], V[1:]
    ok = Vp > 1e-12
    Vp, Vn = Vp[ok], Vn[ok]
    y = (Vn - Vp) / np.sqrt(Vp)
    X = np.column_stack([dt / np.sqrt(Vp), np.sqrt(Vp) * dt])
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    kappa = max(-b, 1e-6)
    eta = max(a / kappa, 1e-10)
    resid = y - X @ np.array([a, b])
    theta = resid.std(ddof=2)
    if scale_theta:
        theta /= np.sqrt(dt)
    return kappa, eta, max(theta, 1e-8)


def estimate(prices, window, q_lo, q_hi, consistent):
    ret = prices.pct_change().dropna()
    n, dt = len(ret), 1.0 / TD
    lo, hi = ret.quantile(q_lo), ret.quantile(q_hi)
    up, dn = ret[ret >= hi], ret[ret <= lo]
    allj = pd.concat([up, dn])

    sd = ret.rolling(window).std().dropna()          # proxy V_t: std harian bergulir
    V = sd**2 * TD if consistent else sd             # konsisten: varians tahunan
    kappa, eta, theta = cir_ols(V.values, dt, scale_theta=consistent)
    x = V.diff().dropna() if consistent else V
    rho = float(np.corrcoef(ret.reindex(x.index), x)[0, 1])

    f = TD if consistent else 1                      # konsisten: parameter tahunan
    nz = lambda v: float(np.nan_to_num(v))
    p = dict(S0=float(prices.iloc[-1]), V0=float(V.iloc[-1]), r=float(ret.mean() * f),
             lam=len(allj) / n * f,
             k_up=nz(up.mean()), k_dn=nz(dn.mean()),
             d_up=nz(up.std(ddof=1)), d_dn=nz(dn.std(ddof=1)),
             kappa=kappa, eta=eta, theta=theta, rho=rho,
             k_trans=None if consistent else nz(allj.mean()),
             d_trans=None if consistent else nz(allj.std(ddof=1)))
    return p, ret, up, dn, (lo, hi)


# ------------------------------------------------------ MONTE CARLO
def simulate(p, steps, n_paths, seed, keep=15):
    rng = np.random.default_rng(seed)
    dt = 1.0 / TD
    mu_u = np.log1p(p["k_up"]) - 0.5 * p["d_up"] ** 2
    mu_d = np.log1p(p["k_dn"]) - 0.5 * p["d_dn"] ** 2
    kbar = 0.5 * (p["k_up"] + p["k_dn"])
    S = np.full(n_paths, p["S0"]); V = np.full(n_paths, p["V0"])
    peak, mdd = S.copy(), np.zeros(n_paths)
    paths = [S[:keep].copy()]
    for _ in range(steps):
        z1 = rng.standard_normal(n_paths)
        z2 = p["rho"] * z1 + np.sqrt(1 - p["rho"] ** 2) * rng.standard_normal(n_paths)
        Vp = np.maximum(V, 0)
        V = V + p["kappa"] * (p["eta"] - Vp) * dt + p["theta"] * np.sqrt(Vp * dt) * z2
        Vp = np.maximum(V, 0)
        N = rng.poisson(p["lam"] * dt, n_paths)
        nu = rng.binomial(N, 0.5); nd = N - nu
        J = nu * mu_u + nd * mu_d + np.sqrt(nu * p["d_up"] ** 2 + nd * p["d_dn"] ** 2) * rng.standard_normal(n_paths)
        S = S * np.exp((p["r"] - p["lam"] * kbar - 0.5 * Vp) * dt + np.sqrt(Vp * dt) * z1 + J)
        peak = np.maximum(peak, S)
        mdd = np.maximum(mdd, 1 - S / peak)
        paths.append(S[:keep].copy())
    return S, mdd, np.array(paths)


# ------------------------------------------------------------- TDF
def bates_cf(phi, x0, T, p, jumps):
    phik, kbar = 0, 0
    for w, kh, d in jumps:
        m = np.log1p(kh) - 0.5 * d**2
        phik = phik + w * np.exp(1j * m * phi - 0.5 * d**2 * phi**2)
        kbar += w * kh
    PhiJ = np.exp(p["lam"] * T * (phik - 1.0))
    k, e, th, rho = p["kappa"], p["eta"], p["theta"], p["rho"]
    b = k - rho * th * 1j * phi
    d = np.sqrt(b**2 + th**2 * (1j * phi + phi**2))
    g = (b - d) / (b + d)
    edT = np.exp(-d * T)
    C = 1j * phi * (p["r"] - p["lam"] * kbar) * T + k * e / th**2 * ((b - d) * T - 2 * np.log((1 - g * edT) / (1 - g)))
    D = (b - d) / th**2 * (1 - edT) / (1 - g * edT)
    return np.exp(1j * phi * x0 + C + D * p["V0"]) * PhiJ


def tdf(p, T, phi_max):
    x0 = np.log(p["S0"])
    if p.get("k_trans") is not None:
        jumps = [(1.0, p["k_trans"], p["d_trans"])]
    else:
        jumps = [(0.5, p["k_up"], p["d_up"]), (0.5, p["k_dn"], p["d_dn"])]
    jm = max(abs(p["k_up"]), abs(p["k_dn"]), p["d_up"], p["d_dn"], p.get("d_trans") or 0)
    sd = np.sqrt(max(p["V0"], p["eta"]) * T + p["lam"] * T * jm**2) + 1e-6
    inf = np.isinf(phi_max)
    x = np.linspace(x0 - 8 * sd - 0.3, x0 + 8 * sd + 0.3, 800 if inf else 1200)
    if inf:
        f = lambda ph: np.real(np.exp(-1j * x * ph) * bates_cf(ph, x0, T, p, jumps))
        val, _ = quad_vec(f, 0.0, np.inf, epsabs=1e-12, epsrel=1e-9, limit=2000)
    else:
        pe = min(max(phi_max, 12 / sd), 600.0)        # batas efektif, menyesuaikan sebaran
        phi = np.linspace(1e-8, pe, 2500)
        Phi = bates_cf(phi, x0, T, p, jumps)
        val = np.trapezoid(np.real(np.exp(-1j * np.outer(x, phi)) * Phi[None, :]), phi, axis=1)
    px = np.clip(val / np.pi, 0, None)
    S = np.exp(x)
    pS = px / S
    return S, pS / np.trapezoid(pS, S)


def dens_stats(S, pdf, qlo, qhi):
    m = np.trapezoid(S * pdf, S); sd = np.sqrt(np.trapezoid((S - m) ** 2 * pdf, S))
    cdf = np.cumsum(pdf * np.gradient(S)); cdf /= cdf[-1]
    return dict(Mean=m, Std=sd, Skew=np.trapezoid(((S - m) / sd) ** 3 * pdf, S),
                Kurt=np.trapezoid(((S - m) / sd) ** 4 * pdf, S) - 3,
                Qlo=np.interp(qlo, cdf, S), Qhi=np.interp(qhi, cdf, S)), cdf


def samp_stats(x, qlo, qhi):
    return dict(Mean=x.mean(), Std=x.std(ddof=1), Skew=stats.skew(x), Kurt=stats.kurtosis(x),
                Qlo=np.quantile(x, qlo), Qhi=np.quantile(x, qhi))


# -------------------------------------------------------------- UI
def main():
    st.set_page_config(page_title="Bates Risk Analyzer", layout="wide")
    st.title("Analisis Risiko Harga Saham - Model Bates")
    st.caption("Stochastic volatility + jumps. Hasil berupa distribusi harga ke depan, bukan prediksi dan bukan saran investasi.")

    with st.sidebar:
        st.header("Input")
        ticker = st.text_input("Ticker (Yahoo Finance)", "ANTM.JK").strip().upper()
        period = st.selectbox("Periode data", ["1y", "3y", "5y", "10y", "max"], index=2)
        horizon = st.select_slider("Horizon (hari kerja)", [21, 63, 126, 252], value=252)
        consistent = st.radio("Mode parameter", ["Konsisten (tahunan)", "Replikasi skripsi"]).startswith("Konsisten")
        with st.expander("Pengaturan lanjutan"):
            window = st.slider("Window volatilitas (hari)", 5, 60, 20)
            q_lo_j = st.slider("Persentil jump turun", 1, 10, 5) / 100
            q_hi_j = st.slider("Persentil jump naik", 90, 99, 95) / 100
            n_paths = st.select_slider("Jumlah simulasi", [1_000, 10_000, 50_000], value=10_000)
            phi_inf = st.checkbox("Integral TDF -inf..inf (lebih lambat)", False)
            phi_max = np.inf if phi_inf else st.number_input("Batas integral phi", 20.0, 600.0, 100.0)
            seed = st.number_input("Seed", 0, 10_000, 42)
        drop = st.slider("Probabilitas turun lebih dari (%)", 1, 50, 10)
        run = st.button("Jalankan analisis", type="primary", use_container_width=True)

    if run:
        with st.spinner("Mengambil data..."):
            prices = load_prices(ticker, period)
        if prices is None or len(prices) < window + 30:
            st.error("Data tidak ditemukan atau terlalu sedikit. Periksa ticker (saham IDX: akhiran .JK).")
            return
        p, ret, up, dn, (lo, hi) = estimate(prices, window, q_lo_j, q_hi_j, consistent)
        with st.spinner("Simulasi dan transisi densitas..."):
            ST, mdd, paths = simulate(p, horizon, n_paths, seed)
            S, pS = tdf(p, horizon / TD, phi_max)
        st.session_state["res"] = dict(prices=prices, p=p, ret=ret, up=up, dn=dn, ST=ST, mdd=mdd,
                                       paths=paths, S=S, pS=pS, drop=drop, horizon=horizon,
                                       consistent=consistent, ticker=ticker, n=len(prices))
    r = st.session_state.get("res")
    if r is None:
        st.info("Pilih saham dan periode di sidebar, lalu klik **Jalankan analisis**.")
        return

    p, ST, S, pS = r["p"], r["ST"], r["S"], r["pS"]
    ql, qh = 0.05, 0.95
    mc = samp_stats(ST, ql, qh)
    td, cdf = dens_stats(S, pS, ql, qh)
    level = p["S0"] * (1 - r["drop"] / 100)
    prob_mc, prob_td = float((ST < level).mean()), float(np.interp(level, S, cdf))

    # peringatan
    if p["kappa"] / TD > 1:
        st.warning(f"kappa*dt = {p['kappa']/TD:.2f} > 1: Euler kurang stabil, volatilitas simulasi akan berosilasi.")
    if abs(td["Std"] / mc["Std"] - 1) > 0.1:
        st.warning("TDF dan Monte Carlo berbeda lebih dari 10% pada std. Coba perbesar batas phi atau jumlah simulasi.")
    if r["n"] > 2000:
        st.warning("Data sangat panjang: parameter bisa berubah antar rezim pasar. Bandingkan dengan periode lebih pendek.")
    if not r["consistent"]:
        st.warning("Mode replikasi: parameter harian dipakai sebagai nilai tahunan (seperti skripsi), "
                   "sehingga intensitas jump sangat rendah.")

    st.subheader(f"{r['ticker']} - horizon {r['horizon']} hari kerja")
    c = st.columns(5)
    c[0].metric("Harga terakhir", f"{p['S0']:,.0f}")
    c[1].metric("Rata-rata harga akhir (TDF)", f"{td['Mean']:,.0f}")
    c[2].metric("Batas bawah P5 (VaR 5%)", f"{td['Qlo']:,.0f}", f"{td['Qlo']/p['S0']-1:.1%}")
    c[3].metric("Batas atas P95", f"{td['Qhi']:,.0f}", f"{td['Qhi']/p['S0']-1:.1%}")
    c[4].metric(f"P(turun > {r['drop']}%)", f"{prob_td:.1%}", f"MC: {prob_mc:.1%}", delta_color="off")

    t1, t2, t3 = st.tabs(["Distribusi", "Data dan jump", "Parameter dan statistik"])
    with t1:
        a, b = st.columns(2)
        fig = Figure(figsize=(6, 4)); ax = fig.subplots()
        ax.hist(ST, bins=80, density=True, alpha=.35, label="Monte Carlo")
        ax.plot(S, pS, "g", label="TDF")
        for q, col in ((td["Qlo"], "r"), (td["Qhi"], "b")):
            ax.axvline(q, color=col, ls="--", lw=1)
        ax.axvline(p["S0"], color="k", lw=1)
        ax.set_xlim(max(0, np.quantile(ST, .001)), np.quantile(ST, .999))
        ax.set_xlabel("Harga akhir"); ax.set_ylabel("Densitas"); ax.legend(); ax.set_title("Distribusi harga akhir")
        a.pyplot(fig)
        fig = Figure(figsize=(6, 4)); ax = fig.subplots()
        ax.plot(r["paths"], lw=.7); ax.set_xlabel("Hari"); ax.set_ylabel("Harga"); ax.set_title("Contoh lintasan simulasi")
        b.pyplot(fig)
        a, b = st.columns(2)
        fig = Figure(figsize=(6, 4)); ax = fig.subplots()
        ax.hist(r["mdd"] * 100, bins=60, color="tab:red", alpha=.6)
        ax.axvline(np.quantile(r["mdd"], .95) * 100, color="k", ls="--")
        ax.set_xlabel("Max drawdown (%)"); ax.set_title("Max drawdown per lintasan (garis: P95)")
        a.pyplot(fig)
        b.metric("Max drawdown tipikal (median)", f"{np.median(r['mdd']):.1%}")
        b.metric("Max drawdown buruk (P95)", f"{np.quantile(r['mdd'], .95):.1%}")
    with t2:
        fig = Figure(figsize=(11, 3.8)); ax = fig.subplots()
        ax.plot(r["prices"].index, r["prices"].values, lw=1)
        ax.scatter(r["prices"].reindex(r["up"].index).index, r["prices"].reindex(r["up"].index), c="g", s=14, label="Jump naik")
        ax.scatter(r["prices"].reindex(r["dn"].index).index, r["prices"].reindex(r["dn"].index), c="r", s=14, label="Jump turun")
        ax.legend(); ax.set_title(f"Harga {r['ticker']} (disesuaikan) dan jump terdeteksi")
        st.pyplot(fig)
        st.caption(f"{len(r['up'])} jump naik dan {len(r['dn'])} jump turun dari {r['n']-1} return harian.")
    with t3:
        a, b = st.columns(2)
        names = {"S0": "S0", "V0": "V0", "r": "r", "lam": "lambda", "k_up": "k naik", "k_dn": "k turun",
                 "d_up": "delta naik", "d_dn": "delta turun", "kappa": "kappa", "eta": "eta",
                 "theta": "theta", "rho": "rho", "k_trans": "k transisi", "d_trans": "delta transisi"}
        a.dataframe(pd.DataFrame({"Parameter": [names[k] for k in p if p[k] is not None],
                                  "Nilai": [p[k] for k in p if p[k] is not None]}), hide_index=True)
        tab = pd.DataFrame([mc, td], index=["Monte Carlo", "TDF"]).rename(
            columns={"Qlo": "P5", "Qhi": "P95"}).round(2)
        b.dataframe(tab)
        b.caption("Varians awal V0 dan deret V_t diproksi dari standar deviasi return harian bergulir.")


if __name__ == "__main__":
    main()
