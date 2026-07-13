# -*- coding: utf-8 -*-
"""
Projeção de Receitas de Petróleo/Gás e do Fundo Soberano — Estado do RJ
=======================================================================

App Streamlit com duas abas:

  1) 🛢️ Petróleo e Gás — projeção de Royalties + Participação Especial do RJ
     a partir da previsão de produção da ANP, da curva a termo do Brent
     (Yahoo/TradingView) e do Henry Hub para o gás.

  2) 🏦 Fundo Soberano — projeção da capitalização do fundo estadual, com o
     patrimônio aplicado em títulos públicos de curto/médio prazo (≤ 4 anos)
     remunerados pela Selic, e saque de parte dos rendimentos.

Fontes de dados no diretório:
  * previsao-producao.csv  ........ Previsão de produção da ANP (m³/ano por bacia)
  * royalties_calculo.xlsx ........ Metodologia de cálculo (repartição RJ)
  * serie_historica_receitas ....... calibra a projeção linear da Part. Especial

Metodologia de repartição (regra vigente — liminar STF 2013), conforme a planilha:
  - Cota RJ na parcela de 5% (mínima) ........ 30%   (Lei 7.990/89, art. 48)
  - Cota RJ na parcela excedente (>5%) ....... 22,5% (Lei 9.478/97, art. 49, II)
  - Cota RJ na Participação Especial ......... 40%   (Lei 9.478/97, art. 50)
"""

import io
import math
import datetime as dt
from statistics import NormalDist

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
M3_TO_BBL = 6.28981           # 1 m³ de petróleo = 6,28981 barris (fator ANP)
GAS_M3_TO_MMBTU = 0.0359      # ~37,9 MJ/m³ ÷ 1.055 MJ/MMBtu (poder calorífico típico)
BACIAS_RJ_PADRAO = ["Campos", "Santos"]   # produção no mar atribuída ao RJ
MESES_CODE = "FGHJKMNQUVXZ"   # códigos de mês dos futuros: F=Jan ... Z=Dez

COTA_RJ_5PCT_DEFAULT = 0.30
COTA_RJ_EXC_DEFAULT = 0.225
COTA_RJ_PE_DEFAULT = 0.40

PATRIMONIO_FUNDO_DEFAULT = 2.045   # R$ bilhões

# Cenários de preço (dimensionados por volatilidade implícita)
CENARIO_CORES = {"Baixo": "#d62728", "Base": "#1f77b4", "Alto": "#2ca02c"}
CONF_NIVEIS = {                       # rótulo -> quantil superior (z = inv_cdf)
    "P10–P90 (80% de confiança)": 0.90,
    "P05–P95 (90% de confiança)": 0.95,
    "P25–P75 (50% de confiança)": 0.75,
}
FAN_QUANTIS = [0.10, 0.25, 0.50, 0.75, 0.90]   # bandas do fan chart

PROD_CSV = "previsao-producao.csv"
HIST_CSV = "serie_historica_receitas_2015-2026.csv"

st.set_page_config(
    page_title="Projeção RJ — Petróleo, Gás e Fundo Soberano",
    page_icon="🛢️",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Carregamento de dados de produção / histórico
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def carregar_producao(path: str = PROD_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",", decimal=",", thousands=".", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=True)
def carregar_historico_pe_royalties(path: str = HIST_CSV):
    """Agrega PE e Royalties realizados pelo RJ por ano (calibra a projeção linear)."""
    try:
        df = pd.read_csv(
            path, sep=";", decimal=",", encoding="utf-8-sig",
            usecols=["Ano", "Natureza da Receita (Tit. NR)", "Receita Realizada (mês)"],
        )
    except Exception:
        return None, False
    df["val"] = pd.to_numeric(df["Receita Realizada (mês)"], errors="coerce").fillna(0.0)
    nat = df["Natureza da Receita (Tit. NR)"].fillna("").str.lower()
    is_ded = nat.str.contains("dedu")
    is_pe = nat.str.contains("participa") & nat.str.contains("espec")
    is_roy = nat.str.contains("royal")

    def por_ano(mask):
        return df[mask].groupby("Ano")["val"].sum()

    pe = por_ano(is_pe & ~is_ded).add(por_ano(is_pe & is_ded), fill_value=0)
    roy = por_ano(is_roy & ~is_ded).add(por_ano(is_roy & is_ded), fill_value=0)
    out = pd.DataFrame({"PE_liquida": pe, "Royalties_liquido": roy}).dropna()
    out = out[(out["PE_liquida"] > 0) & (out["Royalties_liquido"] > 0)]
    return out, True


# --------------------------------------------------------------------------- #
# Preços — Brent (curva a termo), câmbio e Henry Hub
# --------------------------------------------------------------------------- #
def _close_series(d):
    c = d["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c


@st.cache_data(show_spinner=True, ttl=60 * 60)
def baixar_front_e_cambio(lookback_meses: int = 24):
    """BZ=F (Brent front), USDBRL=X e NG=F (Henry Hub) — médias mensais."""
    try:
        import yfinance as yf
    except Exception as e:
        return None, f"yfinance indisponível: {e}"
    try:
        fim = dt.date.today()
        ini = fim - dt.timedelta(days=int(lookback_meses * 31) + 5)
        out = {}
        for tk, nome in [("BZ=F", "brent_usd"), ("USDBRL=X", "usdbrl"), ("NG=F", "hh_usd")]:
            d = yf.download(tk, start=ini.isoformat(), end=fim.isoformat(),
                            interval="1d", progress=False, auto_adjust=False)
            if d is not None and not d.empty:
                out[nome] = _close_series(d).resample("MS").mean()
        if "brent_usd" not in out:
            return None, "Sem dados de Brent (BZ=F) no Yahoo Finance."
        m = pd.DataFrame(out)
        for c in ("usdbrl", "hh_usd"):
            if c in m:
                m[c] = m[c].ffill().bfill()
        m.index = m.index.to_period("M").to_timestamp()
        return m.dropna(subset=["brent_usd"]), None
    except Exception as e:
        return None, f"Falha no Yahoo Finance: {e}"


@st.cache_data(show_spinner=True, ttl=60 * 60)
def baixar_curva_brent_yahoo(ano_ini: int, ano_fim: int):
    """
    Curva a termo do Brent a partir dos contratos futuros mensais do Yahoo
    (BZ{mês}{ano}.NYM). Retorna DataFrame [ano, mes, fwd_usd] com o último preço
    de cada contrato disponível.
    """
    try:
        import yfinance as yf
    except Exception as e:
        return None, f"yfinance indisponível: {e}"
    try:
        tickers = [f"BZ{mc}{yy % 100:02d}.NYM"
                   for yy in range(ano_ini, ano_fim + 1) for mc in MESES_CODE]
        data = yf.download(" ".join(tickers), period="10d",
                           progress=False, auto_adjust=False, group_by="ticker")
        rows = []
        for yy in range(ano_ini, ano_fim + 1):
            for i, mc in enumerate(MESES_CODE):
                tk = f"BZ{mc}{yy % 100:02d}.NYM"
                try:
                    col = data[tk]["Close"] if isinstance(data.columns, pd.MultiIndex) else data["Close"]
                    val = col.dropna()
                    if len(val):
                        rows.append((yy, i + 1, float(val.iloc[-1])))
                except Exception:
                    continue
        if not rows:
            return pd.DataFrame(columns=["ano", "mes", "fwd_usd"]), "Nenhum contrato futuro do Brent retornado."
        return pd.DataFrame(rows, columns=["ano", "mes", "fwd_usd"]), None
    except Exception as e:
        return None, f"Falha ao montar a curva do Brent: {e}"


@st.cache_data(show_spinner=True, ttl=60 * 60)
def baixar_curva_brent_tradingview(n_contratos: int = 18):
    """
    Fallback: curva a termo via TradingView (BRN1!..BRNn!) usando a lib opcional
    tvDatafeed. Requer instalação manual:
        pip install --upgrade git+https://github.com/rongardF/tvdatafeed.git
    Retorna DataFrame [ano, mes, fwd_usd] ou (None, motivo).
    """
    try:
        from tvDatafeed import TvDatafeed, Interval
    except Exception:
        return None, ("tvDatafeed não instalado — fallback do TradingView indisponível "
                      "(instale via git para habilitar BRN1!).")
    try:
        from dateutil.relativedelta import relativedelta
        tv = TvDatafeed()
        base = dt.date.today().replace(day=1)
        rows = []
        for k in range(1, n_contratos + 1):
            df = tv.get_hist(symbol=f"BRN{k}!", exchange="ICEEUR",
                             interval=Interval.in_daily, n_bars=5)
            if df is not None and not df.empty:
                deliv = base + relativedelta(months=k)  # aproximação do vencimento
                rows.append((deliv.year, deliv.month, float(df["close"].iloc[-1])))
        if not rows:
            return None, "TradingView não retornou contratos BRN."
        return pd.DataFrame(rows, columns=["ano", "mes", "fwd_usd"]), None
    except Exception as e:
        return None, f"Falha no TradingView: {e}"


def preco_usd_anual(anos, stat, curva_df, front_usd_stat):
    """
    Consolida a curva a termo em um preço anual (USD/bbl) por ano:
      - anos presentes na curva: média/mediana dos preços mensais forward do ano;
      - anos anteriores ao início da curva (já decorridos): preço front (BZ=F);
      - anos além da curva: carrega o último forward anual conhecido.
    """
    out = {a: None for a in anos}
    if curva_df is not None and not curva_df.empty:
        for a in anos:
            vals = curva_df[curva_df["ano"] == a]["fwd_usd"]
            if len(vals):
                out[a] = float(vals.mean() if stat == "media" else vals.median())
    conhecidos = sorted([a for a in anos if out[a] is not None])
    for a in anos:
        if out[a] is None:
            if not conhecidos:
                out[a] = front_usd_stat
            elif a < min(conhecidos):
                out[a] = front_usd_stat
            else:
                out[a] = out[max(conhecidos)]
    origem = {a: ("curva a termo" if (curva_df is not None and not curva_df.empty
                 and a in set(curva_df["ano"])) else
                 ("front BZ=F" if not conhecidos or a < min(conhecidos) else "extrapolado"))
              for a in anos}
    return pd.Series(out, name="preco_usd"), origem


@st.cache_data(show_spinner=True, ttl=60 * 60)
def baixar_volatilidades(lookback_meses: int = 24):
    """
    Volatilidade para dimensionar as bandas de cenário:
      - petróleo: volatilidade IMPLÍCITA via índice OVX da CBOE (^OVX);
      - gás: volatilidade histórica anualizada de NG=F (não há índice de vol.
        implícita de gás natural com série pública livre);
      - petróleo (histórica, BZ=F): fallback caso o OVX não retorne.
    """
    info = {"oil_iv": None, "oil_hist": None, "gas_hist": None,
            "fonte_oleo": None, "fonte_gas": None, "err": None}
    try:
        import yfinance as yf
    except Exception as e:
        info["err"] = f"yfinance indisponível: {e}"
        return info
    try:
        fim = dt.date.today()
        ini = fim - dt.timedelta(days=int(lookback_meses * 31) + 5)

        def hist_vol(tk):
            d = yf.download(tk, start=ini.isoformat(), end=fim.isoformat(),
                            interval="1d", progress=False, auto_adjust=False)
            if d is None or d.empty:
                return None
            r = np.log(_close_series(d).dropna()).diff().dropna()
            return float(r.std() * np.sqrt(252)) if len(r) > 5 else None

        ov = yf.download("^OVX", start=ini.isoformat(), end=fim.isoformat(),
                         interval="1d", progress=False, auto_adjust=False)
        if ov is not None and not ov.empty:
            info["oil_iv"] = float(_close_series(ov).dropna().iloc[-1]) / 100.0
            info["fonte_oleo"] = "OVX — vol. implícita (CBOE)"
        info["oil_hist"] = hist_vol("BZ=F")
        info["gas_hist"] = hist_vol("NG=F")
        info["fonte_gas"] = "NG=F — vol. histórica"
    except Exception as e:
        info["err"] = str(e)
    return info


def _horizonte_anos(ano):
    """Horizonte (em anos) de hoje até o ano; 0 para anos já decorridos."""
    h = ano - dt.date.today().year
    return 0.0 if h < 0 else h + 0.5


def fatores_quantil(anos, vol, z):
    """
    Fator multiplicativo lognormal por ano: exp(z·σ·√t).
    z=0 → curva base (fator 1). Bandas crescem com √(horizonte).
    """
    if vol is None or vol <= 0:
        return pd.Series({a: 1.0 for a in anos})
    return pd.Series({a: math.exp(z * vol * math.sqrt(_horizonte_anos(a))) for a in anos})


# --------------------------------------------------------------------------- #
# Núcleo de cálculo (metodologia da planilha)
# --------------------------------------------------------------------------- #
def volume_por_ano(df_prod, bacias, ambiente, fator, coluna):
    sub = df_prod[df_prod["BACIA"].isin(bacias)]
    if ambiente and ambiente != "TODOS":
        sub = sub[sub["AMBIENTE"] == ambiente]
    return (sub.groupby("ANO")[coluna].sum() * fator).rename(coluna)


def calcular_royalties(valor_producao, aliquota, cota_5, cota_exc):
    parcela_min = 0.05 * valor_producao
    parcela_exc = np.maximum(aliquota - 0.05, 0.0) * valor_producao
    rj_5 = parcela_min * cota_5
    rj_exc = parcela_exc * cota_exc
    return pd.DataFrame({
        "valor_producao": valor_producao,
        "royalty_total": valor_producao * aliquota,
        "parcela_minima_5pct": parcela_min,
        "parcela_excedente": parcela_exc,
        "rj_parcela_5pct": rj_5,
        "rj_parcela_excedente": rj_exc,
        "royalties_rj": rj_5 + rj_exc,
    })


def pe_por_deducoes(valor_producao, ded_pct_total, aliquota_pe, cota_pe):
    receita_liquida = valor_producao * (1 - ded_pct_total)
    pe_campo = receita_liquida * aliquota_pe
    return (pe_campo * cota_pe).rename("pe_rj")


def ajustar_pe_linear(hist_df):
    x = hist_df["Royalties_liquido"].values
    y = hist_df["PE_liquida"].values
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2


def projetar_fundo(patrimonio0, taxas_anuais, pct_saque, aportes=None):
    """
    Projeta a capitalização do fundo ano a ano.
      rendimento_bruto = patrimônio_início × Selic
      saque            = rendimento_bruto × pct_saque   (revertido a outros fins)
      reinvestido      = rendimento_bruto − saque
      aporte           = receita de petróleo/gás destinada ao fundo (opcional)
      patrimônio_fim   = patrimônio_início + reinvestido + aporte

    'indice_selic' é o rendimento acumulado (base 100) usando apenas a Selic —
    mostra o sinal e a inclinação da curva de rendimentos, sem valores em R$.
    """
    ano0 = dt.date.today().year
    if aportes is None:
        aportes = [0.0] * len(taxas_anuais)
    rows = []
    patr = patrimonio0
    indice = 100.0
    for i, taxa in enumerate(taxas_anuais):
        rend = patr * taxa
        saque = rend * pct_saque
        reinv = rend - saque
        aporte = aportes[i] if i < len(aportes) else 0.0
        patr_fim = patr + reinv + aporte
        indice *= (1 + taxa)
        rows.append({
            "Ano": ano0 + i,
            "Selic": taxa,
            "patr_inicio": patr,
            "rendimento_bruto": rend,
            "saque": saque,
            "reinvestido": reinv,
            "aporte": aporte,
            "patr_fim": patr_fim,
            "indice_selic": indice,
        })
        patr = patr_fim
    return pd.DataFrame(rows), ano0


# --------------------------------------------------------------------------- #
# ABA 1 — Petróleo e Gás
# --------------------------------------------------------------------------- #
def render_petroleo(df_prod, params):
    p = params
    bacias, ambiente, fator_rj, anos_sel = p["bacias"], p["ambiente"], p["fator_rj"], p["anos_sel"]
    if not bacias:
        st.warning("Selecione ao menos uma bacia na barra lateral para ver a projeção de petróleo e gás.")
        return
    if not anos_sel:
        st.warning("Selecione ao menos um ano na barra lateral.")
        return

    stat, modo_preco = p["stat"], p["modo_preco"]
    anos = sorted(anos_sel)

    # ----- Preços ---------------------------------------------------------- #
    front_df, _ = baixar_front_e_cambio(p["lookback"])
    curva_df, erro_curva = baixar_curva_brent_yahoo(min(anos), max(anos) + 1)
    fonte_curva = "Yahoo Finance (BZ*.NYM)"
    if (curva_df is None or curva_df.empty) and p["usar_tv"]:
        tv_df, erro_tv = baixar_curva_brent_tradingview()
        if tv_df is not None and not tv_df.empty:
            curva_df, fonte_curva = tv_df, "TradingView (BRN1!)"
        else:
            erro_curva = (erro_curva or "") + f" | {erro_tv}"

    if p["cambio_modo"] == "Manual" or front_df is None or \
            "usdbrl" not in (front_df.columns if front_df is not None else []):
        usdbrl = p["cambio_manual"]
    else:
        usdbrl = float(front_df["usdbrl"].dropna().iloc[-1])

    if front_df is not None and not front_df.empty:
        s = front_df["brent_usd"].dropna()
        front_usd_stat = float(s.mean() if stat == "media" else s.median())
    else:
        front_usd_stat = st.number_input("Brent front manual (USD/bbl)", value=75.0, min_value=1.0)

    preco_usd, origem_ano = preco_usd_anual(anos, stat, curva_df, front_usd_stat)
    ano_base = anos[0]
    preco_usd = preco_usd * pd.Series({a: (1 + p["reajuste_oleo"]) ** (a - ano_base) for a in anos})
    preco_oleo_brl = (preco_usd * usdbrl).rename("preco_oleo_brl")

    if p["gas_modo"].startswith("Henry Hub") and front_df is not None and "hh_usd" in front_df:
        hh = front_df["hh_usd"].dropna()
        hh_stat = float(hh.mean() if stat == "media" else hh.median())
        gas_base_brl_m3 = hh_stat * GAS_M3_TO_MMBTU * usdbrl
    else:
        gas_base_brl_m3 = p["gas_preco_manual"]
    preco_gas_brl = pd.Series(
        {a: gas_base_brl_m3 * (1 + p["reajuste_gas"]) ** (a - ano_base) for a in anos},
        name="preco_gas_brl")

    # ----- Volatilidade para os cenários ----------------------------------- #
    vol_info = baixar_volatilidades(p["lookback"])
    if p["vol_manual_on"]:
        oil_vol, gas_vol = p["vol_oleo_manual"], p["vol_gas_manual"]
        fonte_oleo = fonte_gas = "volatilidade manual"
    else:
        oil_vol = vol_info["oil_iv"] or vol_info["oil_hist"]
        gas_vol = vol_info["gas_hist"]
        fonte_oleo = vol_info["fonte_oleo"] or (
            "BZ=F — vol. histórica" if vol_info["oil_hist"] else "indisponível")
        fonte_gas = vol_info["fonte_gas"] or "indisponível"
    z = NormalDist().inv_cdf(CONF_NIVEIS[p["conf_nivel"]])

    # ----- Painel de preços ------------------------------------------------ #
    st.subheader("💵 Preço de referência — curva a termo do Brent")
    c1, c2 = st.columns([2, 1])
    with c2:
        st.metric("Câmbio USD/BRL", f"R$ {usdbrl:,.2f}")
        if oil_vol:
            st.metric("Vol. do óleo (bandas)", f"{oil_vol*100:,.1f}% a.a.")
        if p["gas_modo"].startswith("Henry Hub") and p["incluir_gas"]:
            st.metric("Gás (Henry Hub → R$/m³)", f"R$ {gas_base_brl_m3:,.3f}/m³")
        if erro_curva:
            st.warning(f"Curva a termo: {erro_curva.strip(' |')}")
        st.caption(f"Fonte da curva: **{fonte_curva}**")
        st.caption(f"Vol. óleo: {fonte_oleo}"
                   + (f" · gás: {fonte_gas} ({gas_vol*100:,.0f}%)" if p["incluir_gas"] and gas_vol else ""))
    with c1:
        tab = pd.DataFrame({
            "Preço óleo (US$/bbl)": preco_usd.round(2),
            "Preço óleo (R$/bbl)": preco_oleo_brl.round(2),
            "Origem": pd.Series(origem_ano),
        })
        if p["incluir_gas"]:
            tab["Preço gás (R$/m³)"] = preco_gas_brl.round(3)
        st.dataframe(tab, width='stretch')

    if curva_df is not None and not curva_df.empty:
        fig_c = go.Figure()
        xf = [dt.date(a, 7, 1) for a in anos]

        # Fan chart — bandas de cenário em torno do preço anual (vol. implícita)
        if oil_vol:
            def _banda(q):
                zc = NormalDist().inv_cdf(q)
                return (preco_usd * fatores_quantil(anos, oil_vol, zc)).values
            p10, p25, p75, p90 = _banda(0.10), _banda(0.25), _banda(0.75), _banda(0.90)
            fig_c.add_trace(go.Scatter(x=xf, y=p90, mode="lines", line=dict(width=0),
                                       hoverinfo="skip", showlegend=False))
            fig_c.add_trace(go.Scatter(x=xf, y=p10, mode="lines", line=dict(width=0),
                                       fill="tonexty", fillcolor="rgba(31,119,180,0.12)",
                                       name="Banda P10–P90", hoverinfo="skip"))
            fig_c.add_trace(go.Scatter(x=xf, y=p75, mode="lines", line=dict(width=0),
                                       hoverinfo="skip", showlegend=False))
            fig_c.add_trace(go.Scatter(x=xf, y=p25, mode="lines", line=dict(width=0),
                                       fill="tonexty", fillcolor="rgba(31,119,180,0.22)",
                                       name="Banda P25–P75", hoverinfo="skip"))

        cd = curva_df.copy()
        cd["ref"] = pd.to_datetime(dict(year=cd["ano"], month=cd["mes"], day=1))
        cd = cd.sort_values("ref")
        fig_c.add_trace(go.Scatter(x=cd["ref"], y=cd["fwd_usd"], mode="lines+markers",
                                   name="Forward Brent (US$/bbl)"))
        fig_c.add_trace(go.Scatter(
            x=xf, y=preco_usd.values, mode="lines+markers",
            marker=dict(size=12, symbol="diamond", color="firebrick"),
            line=dict(color="firebrick"), name=f"Preço anual base ({stat})"))
        fig_c.update_layout(
            title="Curva a termo do Brent, preço anual de referência e bandas de cenário",
            height=360, yaxis_title="US$/bbl", margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_c, width='stretch')
        if oil_vol:
            st.caption(f"Bandas (fan chart) dimensionadas pela volatilidade do óleo "
                       f"({oil_vol*100:,.1f}% a.a., {fonte_oleo}); largura ∝ √(horizonte).")

    # ----- Volumes e valor da produção ------------------------------------- #
    vol_oleo_m3 = volume_por_ano(df_prod, bacias, ambiente, fator_rj, "VOLUME PETRÓLEO")
    vol_oleo_m3 = vol_oleo_m3[vol_oleo_m3.index.isin(anos)]
    vol_oleo_bbl = vol_oleo_m3 * M3_TO_BBL

    vol_gas_milm3 = volume_por_ano(df_prod, bacias, ambiente, fator_rj, p["col_gas"])
    vol_gas_milm3 = vol_gas_milm3.reindex(anos).fillna(0.0)
    vol_gas_m3 = vol_gas_milm3 * 1000.0

    # ----- Cenários de preço (base + bandas por volatilidade) -------------- #
    aliquota, cota_5, cota_exc = p["aliquota"], p["cota_5"], p["cota_exc"]
    hist_df, hist_ok = carregar_historico_pe_royalties()
    pe_lin = (p["metodo_pe"] != "Deduções informadas"
              and hist_ok and hist_df is not None and len(hist_df) >= 3)
    if pe_lin:
        a_, b_, r2 = ajustar_pe_linear(hist_df)

    def calc_pe(valor_prod, roy_rj):
        if p["metodo_pe"] == "Deduções informadas":
            return pe_por_deducoes(valor_prod, p["ded_pct"], p["aliquota_pe"], p["cota_pe"])
        if pe_lin:
            return (a_ + b_ * roy_rj).clip(lower=0)
        return roy_rj * 1.0

    def pipeline(preco_oleo_s, preco_gas_s):
        v_oleo = (vol_oleo_bbl * preco_oleo_s).rename("valor_oleo")
        v_gas = (vol_gas_m3 * preco_gas_s).rename("valor_gas") if p["incluir_gas"] \
            else pd.Series(0.0, index=anos, name="valor_gas")
        v_prod = v_oleo.add(v_gas, fill_value=0)
        r_rj = calcular_royalties(v_prod, aliquota, cota_5, cota_exc)["royalties_rj"]
        r_oleo = calcular_royalties(v_oleo, aliquota, cota_5, cota_exc)["royalties_rj"]
        pe = calc_pe(v_prod, r_rj)
        return {"valor_oleo": v_oleo, "valor_gas": v_gas, "valor_producao": v_prod,
                "roy_rj": r_rj, "roy_oleo": r_oleo, "pe_rj": pe, "total": r_rj + pe}

    cenarios = {}
    for nome, zc in [("Baixo", -z), ("Base", 0.0), ("Alto", z)]:
        po = preco_oleo_brl * fatores_quantil(anos, oil_vol, zc)
        pg = preco_gas_brl * fatores_quantil(anos, gas_vol, zc)
        cenarios[nome] = pipeline(po, pg)

    base = cenarios["Base"]
    valor_oleo, valor_gas = base["valor_oleo"], base["valor_gas"]
    valor_producao = base["valor_producao"]
    roy_rj, roy_oleo = base["roy_rj"], base["roy_oleo"]
    roy_gas = roy_rj - roy_oleo
    pe_rj = base["pe_rj"]
    cenarios_tot = pd.DataFrame(
        {nome: cenarios[nome]["total"] for nome in ["Baixo", "Base", "Alto"]})

    if p["metodo_pe"] == "Deduções informadas":
        pe_info = (f"PE por deduções: receita líquida = bruta × (1 − {p['ded_pct']:.0%}); "
                   f"alíquota efetiva {p['aliquota_pe']:.0%}; cota RJ {p['cota_pe']:.0%}.")
    elif pe_lin:
        pe_info = (f"Projeção linear PE = {a_/1e9:,.2f} + {b_:,.3f} × Royalties "
                   f"(calibrada {hist_df.index.min()}–{hist_df.index.max()}, R² = {r2:.2f}). "
                   "Deduções por campo indisponíveis na ANP → projeção linear simples.")
    else:
        pe_info = "Histórico indisponível — razão PE/Royalties = 1,0 (aproximação linear)."

    # ----- Resultado (cenário base) ---------------------------------------- #
    res = pd.DataFrame({
        "Ano": anos,
        "Volume óleo (Mbbl)": (vol_oleo_bbl / 1e6).round(2).values,
        "Volume gás (MM m³)": (vol_gas_m3 / 1e6).round(1).values,
        "Preço óleo (R$/bbl)": preco_oleo_brl.round(2).values,
        "Valor produção (R$ bi)": (valor_producao / 1e9).round(2).values,
        "Royalties RJ (R$ bi)": (roy_rj / 1e9).round(2).values,
        "Part. Especial RJ (R$ bi)": (pe_rj / 1e9).round(2).values,
    })
    res["Total RJ (R$ bi)"] = (res["Royalties RJ (R$ bi)"] + res["Part. Especial RJ (R$ bi)"]).round(2)

    tot_base = cenarios_tot["Base"].sum() / 1e9
    tot_baixo = cenarios_tot["Baixo"].sum() / 1e9
    tot_alto = cenarios_tot["Alto"].sum() / 1e9

    st.divider()
    st.subheader("📊 Projeção anual de receitas do RJ (cenário base)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Período", f"{anos[0]}–{anos[-1]}")
    k2.metric("Royalties (período)", f"R$ {res['Royalties RJ (R$ bi)'].sum():,.1f} bi")
    k3.metric("Part. Especial (período)", f"R$ {res['Part. Especial RJ (R$ bi)'].sum():,.1f} bi")
    k4.metric("Total RJ base (período)", f"R$ {tot_base:,.1f} bi",
              delta=f"Baixo {tot_baixo:,.1f} · Alto {tot_alto:,.1f}", delta_color="off")

    if p["incluir_gas"]:
        st.caption(f"Do total de royalties, gás responde por ~R$ {(roy_gas.sum()/1e9):,.1f} bi "
                   f"({roy_gas.sum()/max(roy_rj.sum(),1)*100:,.0f}%) no período.")
    st.info(f"ℹ️ Participação Especial — {pe_info}")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=res["Ano"], y=res["Royalties RJ (R$ bi)"], name="Royalties"))
    fig.add_trace(go.Bar(x=res["Ano"], y=res["Part. Especial RJ (R$ bi)"], name="Part. Especial"))
    fig.update_layout(barmode="stack", height=420, yaxis_title="R$ bilhões",
                      title="Receitas projetadas do RJ — cenário base (Royalties + PE)",
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width='stretch')
    st.dataframe(res.set_index("Ano"), width='stretch')

    # ----- Comparação de cenários ------------------------------------------ #
    st.divider()
    st.subheader("🎚️ Cenários de preço — Total RJ (Royalties + PE)")
    st.caption(f"Base = curva a termo. Bandas Alto/Baixo no nível **{p['conf_nivel']}**, "
               "dimensionadas pela volatilidade implícita do óleo (OVX) e histórica do gás. "
               "Os cenários afetam **óleo e gás**.")
    fig_s = go.Figure()
    for nome in ["Baixo", "Base", "Alto"]:
        fig_s.add_trace(go.Bar(
            x=list(cenarios_tot.index), y=(cenarios_tot[nome] / 1e9).round(2),
            name=nome, marker_color=CENARIO_CORES[nome]))
    fig_s.update_layout(barmode="group", height=380, yaxis_title="R$ bilhões",
                        title="Total RJ por cenário e por ano",
                        margin=dict(l=10, r=10, t=50, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    xanchor="right", x=1))
    st.plotly_chart(fig_s, width='stretch')

    cmp = (cenarios_tot[["Baixo", "Base", "Alto"]] / 1e9).round(2)
    cmp.columns = ["Baixo (R$ bi)", "Base (R$ bi)", "Alto (R$ bi)"]
    cmp.index = [str(a) for a in cmp.index]     # índice homogêneo (evita erro Arrow)
    cmp.index.name = "Ano"
    cmp.loc["Período"] = cmp.sum().round(2)
    st.dataframe(cmp, width='stretch')

    csv_buf = io.StringIO()
    res.to_csv(csv_buf, index=False, sep=";", decimal=",")
    st.download_button("⬇️ Baixar projeção (CSV)", data=csv_buf.getvalue().encode("utf-8-sig"),
                       file_name="projecao_receitas_rj.csv", mime="text/csv")

    with st.expander("🔎 Detalhamento óleo × gás (por ano)"):
        det = pd.DataFrame({
            "vol_oleo_bbl": vol_oleo_bbl, "vol_gas_m3": vol_gas_m3,
            "preco_oleo_brl_bbl": preco_oleo_brl, "preco_gas_brl_m3": preco_gas_brl,
            "valor_oleo": valor_oleo, "valor_gas": valor_gas,
            "royalties_oleo": roy_oleo, "royalties_gas": roy_gas,
            "royalties_rj_total": roy_rj, "pe_rj": pe_rj,
        })
        st.dataframe(det.round(0), width='stretch')

    if hist_ok and hist_df is not None:
        with st.expander("📈 Histórico RJ (PE × Royalties) usado na calibração"):
            st.dataframe((hist_df / 1e9).round(2).rename(
                columns={"PE_liquida": "PE (R$ bi)", "Royalties_liquido": "Royalties (R$ bi)"}),
                width='stretch')

    with st.expander("📚 Metodologia e premissas"):
        st.markdown(f"""
**Produção:** ANP/SDP — `previsao-producao.csv`. Óleo em **m³** → barris (1 m³ = {M3_TO_BBL} bbl);
gás em **mil m³** → m³. Bacias: {', '.join(bacias)} · ambiente {ambiente} · atribuição RJ {fator_rj:.0%}.

**Preço do petróleo — curva a termo do Brent:** contratos futuros mensais do Yahoo
(`BZ{{mês}}{{ano}}.NYM`); cada contrato dá o preço forward do respectivo mês de entrega.
O preço anual é a **{modo_preco.lower()}** dos preços mensais forward do ano. Anos já
decorridos usam o front-month `BZ=F`; anos além da curva carregam o último forward.
Fallback quando falta série no Yahoo: **TradingView `BRN1!`** (lib `tvDatafeed`) e entrada manual.
Fonte utilizada nesta execução: **{fonte_curva}**. Câmbio USD/BRL: R$ {usdbrl:,.2f}.

**Gás natural:** {'incluído' if p['incluir_gas'] else 'excluído'}. Preço via
{'Henry Hub (NG=F) convertido — 1 m³ ≈ ' + f'{GAS_M3_TO_MMBTU} MMBtu' if p['gas_modo'].startswith('Henry') else 'entrada manual em R$/m³'}.
O valor da produção de gás soma-se ao do óleo antes do cálculo de royalties/PE.

**Royalties (regra vigente — liminar STF 2013):** valor = volume × preço;
parcela mínima = 5% · excedente = (alíquota − 5%); RJ recebe **{cota_5:.1%}** da mínima
e **{cota_exc:.1%}** da excedente. Alíquota: {aliquota:.1%}.

**Participação Especial:** {pe_info} Cota RJ na PE: {p['cota_pe']:.0%}.

*Fontes legais: Lei 9.478/97 (arts. 47–50), Lei 7.990/89, Decreto 2.705/98, ANP.
Regras pela liminar do STF de 2013; se a Lei 12.734/12 for validada, ajuste as cotas.*
""")

    # Total do RJ (Royalties + PE) por cenário, em R$, para alimentar o fundo
    return cenarios_tot


# --------------------------------------------------------------------------- #
# ABA 2 — Fundo Soberano
# --------------------------------------------------------------------------- #
def render_fundo(cenarios_receita=None):
    st.subheader("🏦 Projeção da capitalização do Fundo Soberano do Estado")
    st.caption(
        "Premissa: todo o patrimônio está aplicado em títulos públicos de curto a médio "
        "prazo (horizonte máximo de 4 anos), remunerados pela taxa Selic. Parte dos "
        "rendimentos é sacada a cada ano; opcionalmente, parte das receitas de "
        "petróleo/gás é aportada ao fundo (por cenário)."
    )

    cfg1, cfg2 = st.columns([1, 1])
    with cfg1:
        patr0_bi = st.number_input(
            "Patrimônio atual do fundo (R$ bilhões)",
            value=PATRIMONIO_FUNDO_DEFAULT, min_value=0.0, step=0.001, format="%.3f",
        )
    with cfg2:
        pct_saque = st.slider(
            "Percentual dos rendimentos sacado por ano (revertido a outros fins)",
            0, 100, 30,
            help="Fração do rendimento anual retirada do fundo; o restante é reinvestido.",
        ) / 100.0

    st.markdown("**Expectativa de taxa Selic anual (% a.a.) — próximos 4 anos**")
    defaults = [11.0, 10.0, 9.5, 9.0]
    cols = st.columns(4)
    ano0 = dt.date.today().year
    taxas = []
    for i, c in enumerate(cols):
        taxas.append(c.number_input(
            f"Ano {i+1} ({ano0 + i})", value=defaults[i],
            min_value=0.0, max_value=50.0, step=0.25, key=f"selic_{i}") / 100.0)

    # ----- Aportes por cenário (receitas de petróleo/gás) ------------------ #
    tem_cenarios = cenarios_receita is not None and not cenarios_receita.empty
    usar_aportes, pct_aporte = False, 0.0
    if tem_cenarios:
        ca1, ca2 = st.columns([1, 1])
        with ca1:
            usar_aportes = st.checkbox(
                "Incorporar aportes das receitas de petróleo/gás (por cenário)", value=True)
        with ca2:
            pct_aporte = st.slider(
                "% da receita do RJ (Royalties + PE) destinada ao fundo",
                0, 100, 10, disabled=not usar_aportes,
                help="Aporte anual = % × Total RJ do respectivo cenário (Baixo/Base/Alto).") / 100.0
    else:
        st.caption("ℹ️ Selecione bacias/anos na aba **Petróleo e Gás** para incorporar "
                   "aportes por cenário ao fundo.")

    fund_years = [ano0 + i for i in range(len(taxas))]

    def aportes_do_cenario(scn):
        if not (usar_aportes and tem_cenarios and scn in cenarios_receita):
            return [0.0] * len(taxas)
        serie = cenarios_receita[scn].reindex(fund_years).fillna(0.0)
        return list(serie.values * pct_aporte)

    dff_scn = {}
    for scn in ["Baixo", "Base", "Alto"]:
        dff_scn[scn], _ = projetar_fundo(patr0_bi * 1e9, taxas, pct_saque, aportes_do_cenario(scn))
    dff = dff_scn["Base"]
    mostra_cenarios = usar_aportes and tem_cenarios

    # ----- Métricas -------------------------------------------------------- #
    patr_final = dff["patr_fim"].iloc[-1]
    total_saque = dff["saque"].sum()
    total_rend = dff["rendimento_bruto"].sum()
    total_aporte = dff["aporte"].sum()

    m1, m2, m3, m4 = st.columns(4)
    if mostra_cenarios:
        pf_lo = dff_scn["Baixo"]["patr_fim"].iloc[-1] / 1e9
        pf_hi = dff_scn["Alto"]["patr_fim"].iloc[-1] / 1e9
        m1.metric("Patrimônio final (base)", f"R$ {patr_final/1e9:,.3f} bi",
                  delta=f"Baixo {pf_lo:,.2f} · Alto {pf_hi:,.2f}", delta_color="off")
    else:
        m1.metric("Patrimônio final", f"R$ {patr_final/1e9:,.3f} bi",
                  delta=f"{(patr_final/(patr0_bi*1e9)-1)*100:,.1f}% no período")
    m2.metric("Rendimento bruto (período)", f"R$ {total_rend/1e9:,.3f} bi")
    m3.metric("Total sacado (período)", f"R$ {total_saque/1e9:,.3f} bi")
    if mostra_cenarios:
        m4.metric("Total aportado (base)", f"R$ {total_aporte/1e9:,.3f} bi")
    else:
        cagr = (patr_final / (patr0_bi * 1e9)) ** (1 / len(taxas)) - 1
        m4.metric("Crescimento médio do patrimônio", f"{cagr*100:,.2f}% a.a.")

    # ----- Gráfico 1: curva de rendimentos (sinal e inclinação, sem R$) ---- #
    st.markdown("##### 📈 Curva de rendimentos (base Selic) — sinal e inclinação")
    st.caption("Não representa valores em R$: mostra o nível/sinal e a inclinação da "
               "remuneração esperada. À esquerda, a Selic anual; à direita, o índice de "
               "rendimento acumulado (base 100).")
    x_idx = [ano0 - 1] + list(dff["Ano"])
    y_idx = [100.0] + list(dff["indice_selic"])
    figA = go.Figure()
    figA.add_trace(go.Scatter(
        x=list(dff["Ano"]), y=(dff["Selic"] * 100).round(2),
        name="Selic esperada (% a.a.)", mode="lines+markers",
        line=dict(color="#1f77b4", width=3)))
    figA.add_trace(go.Scatter(
        x=x_idx, y=[round(v, 1) for v in y_idx],
        name="Rendimento acumulado (base 100)", mode="lines+markers",
        line=dict(color="#2ca02c", width=3, dash="dot"), yaxis="y2"))
    figA.add_hline(y=100, line_dash="dash", line_color="gray",
                   annotation_text="base 100", yref="y2")
    figA.update_layout(
        height=360, margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Selic (% a.a.)", color="#1f77b4"),
        yaxis2=dict(title="Índice acumulado (base 100)", color="#2ca02c",
                    overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(figA, width='stretch')

    tend = "ascendente ↑" if taxas[-1] > taxas[0] else ("descendente ↓" if taxas[-1] < taxas[0] else "estável →")
    st.caption(f"Sinal: rendimento **positivo** (índice acumulado {y_idx[-1]:.1f} > 100). "
               f"Inclinação da curva Selic: **{tend}** "
               f"(de {taxas[0]*100:,.2f}% para {taxas[-1]*100:,.2f}%).")

    # ----- Gráfico 2: capitalização do fundo (R$) -------------------------- #
    st.markdown("##### 💰 Capitalização do fundo ao longo dos anos")
    figB = go.Figure()
    # barras (cenário base) no eixo primário
    figB.add_trace(go.Bar(
        x=dff["Ano"], y=(dff["saque"] / 1e9).round(3),
        name="Saque (revertido)", marker_color="#d62728", opacity=0.7))
    figB.add_trace(go.Bar(
        x=dff["Ano"], y=(dff["reinvestido"] / 1e9).round(3),
        name="Rendimento reinvestido", marker_color="#2ca02c", opacity=0.7))
    if mostra_cenarios and total_aporte > 0:
        figB.add_trace(go.Bar(
            x=dff["Ano"], y=(dff["aporte"] / 1e9).round(3),
            name="Aporte petróleo/gás (base)", marker_color="#9467bd", opacity=0.7))

    def _linha_patr(scn):
        return [round(patr0_bi, 3)] + list((dff_scn[scn]["patr_fim"] / 1e9).round(3))
    x_patr = [ano0 - 1] + list(dff["Ano"])

    if mostra_cenarios:
        # banda Baixo–Alto + linha base (eixo secundário)
        figB.add_trace(go.Scatter(
            x=x_patr, y=_linha_patr("Alto"), mode="lines", yaxis="y2",
            line=dict(color=CENARIO_CORES["Alto"], width=1, dash="dot"),
            name="Patrimônio — Alto"))
        figB.add_trace(go.Scatter(
            x=x_patr, y=_linha_patr("Baixo"), mode="lines", yaxis="y2",
            line=dict(color=CENARIO_CORES["Baixo"], width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(31,119,180,0.12)",
            name="Patrimônio — Baixo"))
        figB.add_trace(go.Scatter(
            x=x_patr, y=_linha_patr("Base"), mode="lines+markers", yaxis="y2",
            line=dict(color=CENARIO_CORES["Base"], width=3),
            name="Patrimônio — Base"))
    else:
        figB.add_trace(go.Scatter(
            x=x_patr, y=_linha_patr("Base"), mode="lines+markers", yaxis="y2",
            name="Patrimônio do fundo (R$ bi)", line=dict(color="#1f77b4", width=3)))

    figB.update_layout(
        barmode="stack", height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Fluxos anuais (R$ bi)", rangemode="tozero"),
        yaxis2=dict(title="Patrimônio do fundo (R$ bi)", color="#1f77b4",
                    overlaying="y", side="right", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(figB, width='stretch')
    if mostra_cenarios:
        st.caption(f"Aportes = {pct_aporte:.0%} da receita do RJ de cada cenário. "
                   "A banda mostra o patrimônio entre os cenários Baixo e Alto; a linha, o Base.")

    # ----- Tabela e download ---------------------------------------------- #
    tabela = pd.DataFrame({
        "Ano": dff["Ano"],
        "Selic (%)": (dff["Selic"] * 100).round(2),
        "Patrimônio início (R$ bi)": (dff["patr_inicio"] / 1e9).round(3),
        "Rendimento bruto (R$ bi)": (dff["rendimento_bruto"] / 1e9).round(3),
        "Saque (R$ bi)": (dff["saque"] / 1e9).round(3),
        "Aporte (R$ bi)": (dff["aporte"] / 1e9).round(3),
        "Patrimônio fim (R$ bi)": (dff["patr_fim"] / 1e9).round(3),
    })
    st.dataframe(tabela.set_index("Ano"), width='stretch')

    if mostra_cenarios:
        st.markdown("**Patrimônio ao fim do ano por cenário (R$ bi)**")
        cmp_f = pd.DataFrame(
            {scn: (dff_scn[scn]["patr_fim"] / 1e9).round(3).values for scn in ["Baixo", "Base", "Alto"]},
            index=fund_years)
        cmp_f.index.name = "Ano"
        st.dataframe(cmp_f, width='stretch')

    csv_buf = io.StringIO()
    tabela.to_csv(csv_buf, index=False, sep=";", decimal=",")
    st.download_button("⬇️ Baixar projeção do fundo (CSV)",
                       data=csv_buf.getvalue().encode("utf-8-sig"),
                       file_name="projecao_fundo_soberano.csv", mime="text/csv")

    with st.expander("📚 Metodologia do fundo"):
        st.markdown(f"""
**Patrimônio inicial:** R$ {patr0_bi:,.3f} bilhões, integralmente aplicado em títulos
públicos de curto a médio prazo (horizonte ≤ 4 anos), remunerados pela **Selic**.

**Dinâmica anual:**
- Rendimento bruto = patrimônio no início do ano × Selic esperada do ano
- Saque = rendimento bruto × {pct_saque:.0%} (revertido a outros fins)
- Reinvestido = rendimento bruto − saque
- Aporte = {pct_aporte:.0%} da receita de petróleo/gás do cenário {'(ativo)' if mostra_cenarios else '(desativado)'}
- Patrimônio ao fim = patrimônio no início + reinvestido + aporte

O **primeiro gráfico** não usa valores em R$: apresenta o nível/sinal e a inclinação da
curva de rendimentos (Selic anual e índice acumulado base 100). O **segundo gráfico**
mostra a capitalização efetiva do fundo, em R$; quando os aportes por cenário estão ativos,
a banda cobre os cenários **Baixo–Alto** e a linha central é o **Base**.

*Modelo simplificado: não considera marcação a mercado, tributação ou inflação.
Os cenários de preço vêm da aba Petróleo e Gás (bandas por volatilidade implícita).*
""")


# --------------------------------------------------------------------------- #
# Interface — cabeçalho, sidebar e abas
# --------------------------------------------------------------------------- #
st.title("🛢️ Projeção de Receitas do Estado do Rio de Janeiro")
st.caption(
    "Petróleo e Gás (Royalties + Participação Especial) · Fundo Soberano estadual · "
    "curva a termo do Brent via Yahoo/TradingView · produção ANP"
)

df_prod = carregar_producao()
anos_disp = sorted(df_prod["ANO"].unique())

with st.sidebar:
    st.header("⚙️ Parâmetros — Petróleo e Gás")
    st.caption("Estes controles afetam a aba **Petróleo e Gás**. "
               "A aba **Fundo Soberano** tem os próprios campos.")

    st.subheader("Produção (ANP)")
    bacias = st.multiselect(
        "Bacias atribuídas ao RJ", options=sorted(df_prod["BACIA"].unique()),
        default=BACIAS_RJ_PADRAO,
        help="A planilha considera a produção no mar das bacias de Campos e Santos.")
    ambiente = st.radio("Ambiente", ["MAR", "TERRA", "TODOS"], horizontal=True, index=0)
    fator_rj = st.slider(
        "Fator de atribuição ao RJ (%)", 0, 100, 100,
        help="Fração da produção atribuída ao território do RJ (a planilha assume 100%).") / 100.0
    anos_sel = st.multiselect("Anos da projeção", options=anos_disp, default=anos_disp)

    st.subheader("Preço do petróleo — Brent (curva a termo)")
    modo_preco = st.radio(
        "Estatística anual do preço",
        ["Média dos preços mensais", "Mediana dos preços mensais"],
        help="Consolida os preços mensais da curva a termo em um preço anual de referência.")
    stat = "media" if modo_preco.startswith("Média") else "mediana"
    lookback = st.slider("Janela do front-month BZ=F (meses)", 6, 60, 24)
    usar_tv = st.checkbox("Usar TradingView (BRN1!) como fallback se faltar série no Yahoo", value=True)
    cambio_modo = st.radio("Câmbio USD→BRL", ["Automático (Yahoo)", "Manual"], horizontal=True)
    cambio_manual = st.number_input("USD/BRL manual", value=5.40, min_value=1.0,
                                    max_value=15.0, step=0.05,
                                    disabled=(cambio_modo != "Manual"))
    reajuste_oleo = st.slider(
        "Reajuste adicional do preço do óleo (% a.a.)", -20.0, 20.0, 0.0, 0.5,
        help="Sobreposição opcional à curva a termo (0 = usa a curva pura).") / 100.0

    st.subheader("Cenários de preço")
    conf_nivel = st.selectbox(
        "Nível de confiança das bandas", list(CONF_NIVEIS.keys()),
        help="Base = curva a termo. Cenários Alto/Baixo dimensionados pela "
             "volatilidade implícita do petróleo (OVX) e histórica do gás.")
    vol_manual_on = st.checkbox("Definir volatilidade manualmente (uso offline)", value=False)
    vol_oleo_manual = st.number_input("Vol. do óleo (% a.a.)", value=35.0, min_value=1.0,
                                      max_value=150.0, step=1.0,
                                      disabled=not vol_manual_on) / 100.0
    vol_gas_manual = st.number_input("Vol. do gás (% a.a.)", value=60.0, min_value=1.0,
                                     max_value=200.0, step=1.0,
                                     disabled=not vol_manual_on) / 100.0

    st.subheader("Gás natural")
    incluir_gas = st.checkbox("Incluir participação do gás", value=True)
    col_gas = st.radio("Volume de gás", ["VOLUME GÁS", "VOLUME GÁS SEM CO2"],
                       help="Volume de gás usado como base (mil m³).")
    gas_modo = st.radio("Preço do gás", ["Henry Hub (Yahoo NG=F)", "Manual (R$/m³)"],
                        disabled=not incluir_gas)
    gas_preco_manual = st.number_input("Preço do gás (R$/m³)", value=0.60, min_value=0.0,
                                       step=0.05, disabled=(gas_modo != "Manual (R$/m³)"))
    reajuste_gas = st.slider("Reajuste do preço do gás (% a.a.)", -20.0, 20.0, 0.0, 0.5,
                             disabled=not incluir_gas) / 100.0

    st.subheader("Royalties")
    aliquota = st.slider("Alíquota de royalty (%)", 5.0, 15.0, 10.0, 0.5,
                         help="10% concessão; mín. 5%; 15% no pré-sal.") / 100.0
    cota_5 = st.number_input("Cota RJ parcela 5% (%)", value=COTA_RJ_5PCT_DEFAULT * 100) / 100
    cota_exc = st.number_input("Cota RJ parcela excedente (%)", value=COTA_RJ_EXC_DEFAULT * 100) / 100

    st.subheader("Participação Especial")
    metodo_pe = st.radio("Método de cálculo da PE",
                         ["Projeção linear (histórico)", "Deduções informadas"],
                         help="Sem deduções por campo, usa-se a projeção linear calibrada "
                              "pela série histórica de receitas do RJ.")
    cota_pe = st.number_input("Cota RJ na PE (%)", value=COTA_RJ_PE_DEFAULT * 100) / 100
    ded_pct = st.slider("Deduções totais (% da receita bruta)", 0.0, 95.0, 40.0, 1.0,
                        disabled=(metodo_pe != "Deduções informadas")) / 100.0
    aliquota_pe = st.slider("Alíquota efetiva de PE (%)", 0.0, 40.0, 40.0, 5.0,
                            disabled=(metodo_pe != "Deduções informadas")) / 100.0

params = dict(
    bacias=bacias, ambiente=ambiente, fator_rj=fator_rj, anos_sel=anos_sel,
    modo_preco=modo_preco, stat=stat, lookback=lookback, usar_tv=usar_tv,
    cambio_modo=cambio_modo, cambio_manual=cambio_manual, reajuste_oleo=reajuste_oleo,
    conf_nivel=conf_nivel, vol_manual_on=vol_manual_on,
    vol_oleo_manual=vol_oleo_manual, vol_gas_manual=vol_gas_manual,
    incluir_gas=incluir_gas, col_gas=col_gas, gas_modo=gas_modo,
    gas_preco_manual=gas_preco_manual, reajuste_gas=reajuste_gas,
    aliquota=aliquota, cota_5=cota_5, cota_exc=cota_exc,
    metodo_pe=metodo_pe, cota_pe=cota_pe, ded_pct=ded_pct, aliquota_pe=aliquota_pe,
)

tab1, tab2 = st.tabs(["🛢️ Petróleo e Gás", "🏦 Fundo Soberano"])
with tab1:
    cenarios_receita = render_petroleo(df_prod, params)
with tab2:
    render_fundo(cenarios_receita)
