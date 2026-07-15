# -*- coding: utf-8 -*-
"""
Projeção de Receitas de Petróleo/Gás e do Fundo Soberano — Estado do RJ
=======================================================================

App Streamlit com duas abas:

  1) 🛢️ Petróleo e Gás — projeção de Royalties + Participação Especial do RJ
     a partir da previsão de produção da ANP, da curva a termo do Brent
     (Yahoo/TradingView) e do Henry Hub para o gás. Cenários "Baixa" e "Alta"
     em torno da curva a termo (cenário Base), dimensionados pela volatilidade
     implícita (OVX).

  2) 🏦 Fundo Soberano — projeção da capitalização do fundo estadual, com o
     patrimônio aplicado em títulos públicos de curto/médio prazo (≤ 4 anos)
     remunerados pela Selic, saque de parte dos rendimentos e projeção nos três
     cenários de receita (Baixa/Base/Alta).

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

# Cenários de preço — Base = curva a termo; bandas por volatilidade implícita
CENARIO_CORES = {"Baixa": "#d62728", "Base": "#1f77b4", "Alta": "#2ca02c"}
CONF_NIVEIS = {                       # rótulo -> quantil superior (z = inv_cdf)
    "P10–P90 (80% de confiança)": 0.90,
    "P05–P95 (90% de confiança)": 0.95,
    "P25–P75 (50% de confiança)": 0.75,
}
DEFAULT_IMPLIED_VOL = 0.35            # fallback se o OVX não estiver disponível

# Aporte ao fundo (LC 200/2022): 30% do INCREMENTO anual positivo da arrecadação
# com Participação Especial + royalties excedentes (>5%). Não é % da receita cheia.
APORTE_PCT_INCREMENTO = 0.30
ANO_ANCORA_REALIZADO = 2025          # ano-base realizado p/ calibrar o nível projetado

PROD_CSV = "previsao-producao.csv"
HIST_CSV = "serie_historica_receitas_2015-2026.csv"
REALIZADO_CSV = "receitas_petroleo_realizado.csv"   # realizado do Tesouro (RJ)

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


@st.cache_data(show_spinner=False)
def carregar_realizado_tesouro(path: str = REALIZADO_CSV):
    """
    Receita realizada de royalties + Participação Especial do petróleo (líquida de
    deduções), por ano, a partir do extrato do Tesouro/RJ. Usada como âncora de
    nível (calibração) e para reconciliar a projeção com o realizado.
    Obs.: o extrato agrega royalties e PE numa mesma sub-alínea (não separa PE
    de excedente), então fornece o TOTAL realizado por ano. Retorna (Series, ok).
    """
    try:
        r = pd.read_csv(path, sep=";", decimal=",", thousands=".", encoding="latin-1")
    except Exception:
        return None, False
    r.columns = [c.strip() for c in r.columns]
    r["ano"] = pd.to_datetime(r["Posição"], format="%d/%m/%Y", errors="coerce").dt.year
    r["val"] = pd.to_numeric(r["Valor Receita Realizada"], errors="coerce").fillna(0.0)
    nm = r["Nome Sub Alinea"].fillna("").str.lower()
    oleo = nm.str.contains("royalt") | nm.str.contains("participa")   # exclui FEP
    serie = r[oleo].groupby("ano")["val"].sum().dropna()              # deduções já negativas
    serie = serie[serie > 0]
    return serie, len(serie) > 0


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
    """BZ=F (Brent front) e NG=F (Henry Hub) — médias mensais."""
    try:
        import yfinance as yf
    except Exception as e:
        return None, f"yfinance indisponível: {e}"
    try:
        fim = dt.date.today()
        ini = fim - dt.timedelta(days=int(lookback_meses * 31) + 5)
        out = {}
        for tk, nome in [("BZ=F", "brent_usd"), ("NG=F", "hh_usd")]:
            d = yf.download(tk, start=ini.isoformat(), end=fim.isoformat(),
                            interval="1d", progress=False, auto_adjust=False)
            if d is not None and not d.empty:
                out[nome] = _close_series(d).resample("MS").mean()
        if "brent_usd" not in out:
            return None, "Sem dados de Brent (BZ=F) no Yahoo Finance."
        m = pd.DataFrame(out)
        if "hh_usd" in m:
            m["hh_usd"] = m["hh_usd"].ffill().bfill()
        m.index = m.index.to_period("M").to_timestamp()
        return m.dropna(subset=["brent_usd"]), None
    except Exception as e:
        return None, f"Falha no Yahoo Finance: {e}"


@st.cache_data(show_spinner=True, ttl=60 * 60 * 6)
def baixar_cambio_ovx_diario(dia: str):
    """
    Câmbio USD/BRL (USDBRL=X) e volatilidade implícita OVX (^OVX) na cotação
    diária mais recente. O argumento 'dia' (data ISO de hoje) entra na chave de
    cache para forçar a atualização a cada novo dia.
    """
    info = {"usdbrl": None, "usdbrl_data": None, "ovx": None, "ovx_data": None, "err": None}
    try:
        import yfinance as yf
    except Exception as e:
        info["err"] = f"yfinance indisponível: {e}"
        return info
    try:
        for tk, ck in [("USDBRL=X", "usdbrl"), ("^OVX", "ovx")]:
            d = yf.download(tk, period="10d", interval="1d",
                            progress=False, auto_adjust=False)
            if d is not None and not d.empty:
                serie = _close_series(d).dropna()
                if len(serie):
                    info[ck] = float(serie.iloc[-1])
                    info[ck + "_data"] = serie.index[-1].date().isoformat()
    except Exception as e:
        info["err"] = str(e)
    return info


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


def _horizonte_anos(ano):
    """Horizonte (em anos) de hoje até o ano; 0 para anos já decorridos."""
    h = ano - dt.date.today().year
    return 0.0 if h < 0 else h + 0.5


def fatores_quantil(anos, vol, z):
    """
    Fator multiplicativo lognormal por ano: exp(z·σ·√t).
    z=0 → cenário base (fator 1). Bandas crescem com √(horizonte).
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
      aporte           = receita de petróleo/gás destinada ao fundo (por cenário)
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
        return None
    if not anos_sel:
        st.warning("Selecione ao menos um ano na barra lateral.")
        return None

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

    fx_ovx = baixar_cambio_ovx_diario(dt.date.today().isoformat())
    if p["cambio_modo"] == "Manual" or fx_ovx["usdbrl"] is None:
        usdbrl, usdbrl_data = p["cambio_manual"], "definido manualmente"
    else:
        usdbrl, usdbrl_data = fx_ovx["usdbrl"], fx_ovx["usdbrl_data"]
    ovx, ovx_data = fx_ovx["ovx"], fx_ovx["ovx_data"]

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

    # ----- Volatilidade implícita e amplitude dos cenários ----------------- #
    vol_impl = (ovx / 100.0) if ovx is not None else DEFAULT_IMPLIED_VOL
    fonte_vol = f"OVX ({ovx_data})" if ovx is not None else f"padrão {DEFAULT_IMPLIED_VOL:.0%}"
    z = NormalDist().inv_cdf(CONF_NIVEIS[p["conf_nivel"]])

    # ----- Painel de preços ------------------------------------------------ #
    st.subheader("💵 Preço de referência — curva a termo do Brent")
    c1, c2 = st.columns([2, 1])
    with c2:
        st.metric("Câmbio USD/BRL (diário)", f"R$ {usdbrl:,.4f}")
        st.caption(f"Cotação de {usdbrl_data}")
        if ovx is not None:
            st.metric("OVX — vol. implícita (diário)", f"{ovx:,.1f}%")
            st.caption(f"OVX de {ovx_data}")
        if p["gas_modo"].startswith("Henry Hub") and p["incluir_gas"]:
            st.metric("Gás (Henry Hub → R$/m³)", f"R$ {gas_base_brl_m3:,.3f}/m³")
        if erro_curva:
            st.warning(f"Curva a termo: {erro_curva.strip(' |')}")
        st.caption(f"Fonte da curva: **{fonte_curva}**")
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
        cd = curva_df.copy()
        cd["ref"] = pd.to_datetime(dict(year=cd["ano"], month=cd["mes"], day=1))
        cd = cd.sort_values("ref")
        fig_c.add_trace(go.Scatter(x=cd["ref"], y=cd["fwd_usd"], mode="lines+markers",
                                   name="Forward Brent (US$/bbl)"))
        fig_c.add_trace(go.Scatter(
            x=[dt.date(a, 7, 1) for a in anos], y=preco_usd.values, mode="markers",
            marker=dict(size=12, symbol="diamond", color="firebrick"),
            name=f"Preço anual base ({stat})"))
        fig_c.update_layout(title="Curva a termo do Brent e preço anual de referência (Base)",
                            height=340, yaxis_title="US$/bbl",
                            margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_c, width='stretch')

    # ----- Volumes --------------------------------------------------------- #
    vol_oleo_m3 = volume_por_ano(df_prod, bacias, ambiente, fator_rj, "VOLUME PETRÓLEO")
    vol_oleo_m3 = vol_oleo_m3[vol_oleo_m3.index.isin(anos)]
    vol_oleo_bbl = vol_oleo_m3 * M3_TO_BBL

    vol_gas_milm3 = volume_por_ano(df_prod, bacias, ambiente, fator_rj, p["col_gas"])
    vol_gas_milm3 = vol_gas_milm3.reindex(anos).fillna(0.0)
    vol_gas_m3 = vol_gas_milm3 * 1000.0

    # ----- Pipeline de receita e cenários ---------------------------------- #
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
        rr = calcular_royalties(v_prod, aliquota, cota_5, cota_exc)
        r_rj = rr["royalties_rj"]
        r_exc = rr["rj_parcela_excedente"]     # royalties excedentes (>5%), cota RJ
        r_oleo = calcular_royalties(v_oleo, aliquota, cota_5, cota_exc)["royalties_rj"]
        pe = calc_pe(v_prod, r_rj)
        # base LC 200/2022 = PE + royalties excedentes (>5%); exclui a parcela de 5%
        return {"valor_oleo": v_oleo, "valor_gas": v_gas, "valor_producao": v_prod,
                "roy_rj": r_rj, "roy_oleo": r_oleo, "pe_rj": pe, "total": r_rj + pe,
                "base_lc": pe + r_exc}

    # Base = curva a termo (z=0); Baixa/Alta escalam óleo e gás pela vol. implícita
    cenarios = {}
    for nome, zc in [("Baixa", -z), ("Base", 0.0), ("Alta", z)]:
        fator = fatores_quantil(anos, vol_impl, zc)
        cenarios[nome] = pipeline(preco_oleo_brl * fator, preco_gas_brl * fator)

    base = cenarios["Base"]
    valor_oleo, valor_gas = base["valor_oleo"], base["valor_gas"]
    valor_producao = base["valor_producao"]
    roy_rj, roy_oleo = base["roy_rj"], base["roy_oleo"]
    roy_gas = roy_rj - roy_oleo
    pe_rj = base["pe_rj"]
    cenarios_tot = pd.DataFrame(
        {nome: cenarios[nome]["total"] for nome in ["Baixa", "Base", "Alta"]})
    cenarios_base_lc = pd.DataFrame(
        {nome: cenarios[nome]["base_lc"] for nome in ["Baixa", "Base", "Alta"]})

    # Calibração ao realizado (âncora de nível): fator = realizado / projetado no
    # ano-âncora. Corrige a superestimação de nível do modelo antes dos incrementos.
    realizado_ser, real_ok = carregar_realizado_tesouro()
    calib, real_ancora = 1.0, None
    if real_ok and ANO_ANCORA_REALIZADO in realizado_ser.index \
            and ANO_ANCORA_REALIZADO in cenarios_tot.index:
        real_ancora = float(realizado_ser.loc[ANO_ANCORA_REALIZADO])
        proj_ancora = float(cenarios_tot.loc[ANO_ANCORA_REALIZADO, "Base"])
        if proj_ancora > 0:
            calib = real_ancora / proj_ancora

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
    tot_baixa = cenarios_tot["Baixa"].sum() / 1e9
    tot_alta = cenarios_tot["Alta"].sum() / 1e9

    st.divider()
    st.subheader("📊 Projeção anual de receitas do RJ (cenário base)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Período", f"{anos[0]}–{anos[-1]}")
    k2.metric("Royalties (período)", f"R$ {res['Royalties RJ (R$ bi)'].sum():,.1f} bi")
    k3.metric("Part. Especial (período)", f"R$ {res['Part. Especial RJ (R$ bi)'].sum():,.1f} bi")
    k4.metric("Total RJ base (período)", f"R$ {tot_base:,.1f} bi",
              delta=f"Baixa {tot_baixa:,.1f} · Alta {tot_alta:,.1f}", delta_color="off")

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

    # ----- Reconciliação com o realizado (Tesouro) ------------------------- #
    if real_ok:
        st.divider()
        st.subheader("🔎 Reconciliação com o realizado do Tesouro")
        anos_rec = sorted(set(cenarios_tot.index) | set(realizado_ser.index))
        rec = pd.DataFrame({
            "Projetado Total (R$ bi)": (cenarios_tot["Base"].reindex(anos_rec) / 1e9),
            "Realizado Tesouro (R$ bi)": (realizado_ser.reindex(anos_rec) / 1e9),
        })
        rec["Δ (proj − real)"] = rec["Projetado Total (R$ bi)"] - rec["Realizado Tesouro (R$ bi)"]
        rec.index = [str(a) for a in rec.index]
        rec.index.name = "Ano"
        cc1, cc2 = st.columns([2, 1])
        with cc2:
            if real_ancora is not None:
                st.metric(f"Realizado {ANO_ANCORA_REALIZADO} (âncora)", f"R$ {real_ancora/1e9:,.2f} bi")
                st.metric("Projetado no ano-âncora",
                          f"R$ {cenarios_tot.loc[ANO_ANCORA_REALIZADO,'Base']/1e9:,.2f} bi")
                st.metric("Fator de calibração", f"{calib:,.3f}",
                          delta=f"modelo superestima {((1/calib)-1)*100:,.0f}%" if calib < 1 else None,
                          delta_color="off")
        with cc1:
            st.caption("O modelo é comparado ao realizado divulgado pelo Tesouro. O fator de "
                       "calibração (realizado ÷ projetado no ano-âncora) alinha o nível antes "
                       "de calcular os incrementos que alimentam o fundo. O extrato agrega "
                       "royalties + PE numa mesma rubrica, então serve como âncora de nível.")
            st.dataframe(rec.round(2), width='stretch')

    # ----- Comparação de cenários (sem fan chart) -------------------------- #
    st.divider()
    st.subheader("🎚️ Cenários de preço — Total RJ (Royalties + PE)")
    st.caption(f"Base = curva a termo. Cenários **Baixa** e **Alta** no nível "
               f"**{p['conf_nivel']}**, dimensionados pela **volatilidade implícita** "
               f"({vol_impl*100:,.1f}% a.a., {fonte_vol}). Os cenários afetam óleo e gás.")
    fig_s = go.Figure()
    for nome in ["Baixa", "Base", "Alta"]:
        fig_s.add_trace(go.Bar(
            x=list(cenarios_tot.index), y=(cenarios_tot[nome] / 1e9).round(2),
            name=nome, marker_color=CENARIO_CORES[nome]))
    fig_s.update_layout(barmode="group", height=380, yaxis_title="R$ bilhões",
                        title="Total RJ por cenário e por ano",
                        margin=dict(l=10, r=10, t=50, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    xanchor="right", x=1))
    st.plotly_chart(fig_s, width='stretch')

    cmp = (cenarios_tot[["Baixa", "Base", "Alta"]] / 1e9).round(2)
    cmp.columns = ["Baixa (R$ bi)", "Base (R$ bi)", "Alta (R$ bi)"]
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
O preço anual é a **{modo_preco.lower()}** dos preços mensais forward do ano. Câmbio USD/BRL
diário ({usdbrl_data}): R$ {usdbrl:,.4f}.

**Cenários:** o **Base** é a curva a termo. **Baixa** e **Alta** aplicam bandas lognormais
`exp(±z·σ·√t)` sobre óleo e gás, com **σ = volatilidade implícita** ({vol_impl*100:,.1f}% a.a.,
{fonte_vol}) e z do nível **{p['conf_nivel']}**. A largura cresce com √(horizonte).

**Gás natural:** {'incluído' if p['incluir_gas'] else 'excluído'}. Preço via
{'Henry Hub (NG=F) convertido — 1 m³ ≈ ' + f'{GAS_M3_TO_MMBTU} MMBtu' if p['gas_modo'].startswith('Henry') else 'entrada manual em R$/m³'}.

**Royalties (regra vigente — liminar STF 2013):** valor = volume × preço;
parcela mínima = 5% · excedente = (alíquota − 5%); RJ recebe **{cota_5:.1%}** da mínima
e **{cota_exc:.1%}** da excedente. Alíquota: {aliquota:.1%}.

**Participação Especial:** {pe_info} Cota RJ na PE: {p['cota_pe']:.0%}.

*Fontes legais: Lei 9.478/97 (arts. 47–50), Lei 7.990/89, Decreto 2.705/98, ANP.
Regras pela liminar do STF de 2013; se a Lei 12.734/12 for validada, ajuste as cotas.*
""")

    # Dados por cenário para alimentar o fundo (aporte = 30% do incremento da base LC)
    return {"total": cenarios_tot, "base_lc": cenarios_base_lc,
            "calib": calib, "real_ok": real_ok, "ano_ancora": ANO_ANCORA_REALIZADO}


# --------------------------------------------------------------------------- #
# ABA 2 — Fundo Soberano
# --------------------------------------------------------------------------- #
def render_fundo(cenarios_receita=None):
    st.subheader("🏦 Projeção da capitalização do Fundo Soberano do Estado")
    st.caption(
        "Premissa: todo o patrimônio está aplicado em títulos públicos de curto a médio "
        "prazo (horizonte máximo de 4 anos), remunerados pela taxa Selic. Parte dos "
        "rendimentos é sacada a cada ano; o aporte segue a **LC 200/2022** — "
        f"**{APORTE_PCT_INCREMENTO:.0%} do incremento anual** da arrecadação com "
        "Participação Especial + royalties excedentes (não um % da receita cheia)."
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

    # ----- Aportes por cenário (LC 200/2022: 30% do incremento anual) ------ #
    tem_cenarios = (cenarios_receita is not None
                    and isinstance(cenarios_receita, dict)
                    and not cenarios_receita["base_lc"].empty)
    fund_years = [ano0 + i for i in range(len(taxas))]
    calib = cenarios_receita["calib"] if tem_cenarios else 1.0
    base_lc_df = cenarios_receita["base_lc"] if tem_cenarios else None

    def aportes_do_cenario(scn):
        """Aporte_t = 30% × max(base_t − base_{t-1}, 0), base = PE + excedente
        calibrada ao realizado. O ano anterior ao 1º ano do fundo ancora o incremento."""
        if not (tem_cenarios and scn in base_lc_df):
            return [0.0] * len(taxas)
        serie = base_lc_df[scn].sort_index() * calib      # base calibrada, série completa
        incremento = serie.diff()                          # preserva o ano-âncora anterior
        aporte = (APORTE_PCT_INCREMENTO * incremento.clip(lower=0))
        return list(aporte.reindex(fund_years).fillna(0.0).values)

    dff_scn = {}
    for scn in ["Baixa", "Base", "Alta"]:
        dff_scn[scn], _ = projetar_fundo(patr0_bi * 1e9, taxas, pct_saque, aportes_do_cenario(scn))
    dff = dff_scn["Base"]

    if not tem_cenarios:
        st.info("ℹ️ Selecione bacias/anos na aba **Petróleo e Gás** para projetar os três "
                "cenários de receita no fundo. Exibindo apenas a dinâmica base (sem aportes).")
    elif calib != 1.0:
        st.caption(f"Base de incremento calibrada ao realizado {cenarios_receita['ano_ancora']} "
                   f"(fator {calib:,.3f}). Aporte = {APORTE_PCT_INCREMENTO:.0%} do incremento "
                   "anual positivo de (PE + royalties excedentes).")

    # ----- Métricas -------------------------------------------------------- #
    patr_final = dff["patr_fim"].iloc[-1]
    total_saque = dff["saque"].sum()
    total_rend = dff["rendimento_bruto"].sum()
    total_aporte = dff["aporte"].sum()
    pf_lo = dff_scn["Baixa"]["patr_fim"].iloc[-1] / 1e9
    pf_hi = dff_scn["Alta"]["patr_fim"].iloc[-1] / 1e9

    m1, m2, m3, m4 = st.columns(4)
    if tem_cenarios:
        m1.metric("Patrimônio final (base)", f"R$ {patr_final/1e9:,.3f} bi",
                  delta=f"Baixa {pf_lo:,.2f} · Alta {pf_hi:,.2f}", delta_color="off")
        m4.metric("Total aportado (base)", f"R$ {total_aporte/1e9:,.3f} bi")
    else:
        cagr = (patr_final / (patr0_bi * 1e9)) ** (1 / len(taxas)) - 1
        m1.metric("Patrimônio final", f"R$ {patr_final/1e9:,.3f} bi",
                  delta=f"{(patr_final/(patr0_bi*1e9)-1)*100:,.1f}% no período")
        m4.metric("Crescimento médio do patrimônio", f"{cagr*100:,.2f}% a.a.")
    m2.metric("Rendimento bruto (período)", f"R$ {total_rend/1e9:,.3f} bi")
    m3.metric("Total sacado (período)", f"R$ {total_saque/1e9:,.3f} bi")

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

    # ----- Gráfico 2: capitalização do fundo por cenário (R$) -------------- #
    st.markdown("##### 💰 Capitalização do fundo ao longo dos anos (por cenário)")
    figB = go.Figure()
    # barras (cenário base) no eixo primário
    figB.add_trace(go.Bar(
        x=dff["Ano"], y=(dff["saque"] / 1e9).round(3),
        name="Saque (revertido)", marker_color="#d62728", opacity=0.6))
    figB.add_trace(go.Bar(
        x=dff["Ano"], y=(dff["reinvestido"] / 1e9).round(3),
        name="Rendimento reinvestido", marker_color="#2ca02c", opacity=0.6))
    if total_aporte > 0:
        figB.add_trace(go.Bar(
            x=dff["Ano"], y=(dff["aporte"] / 1e9).round(3),
            name="Aporte petróleo/gás (base)", marker_color="#9467bd", opacity=0.6))

    # linhas de patrimônio — uma por cenário (sem preenchimento/fan)
    x_patr = [ano0 - 1] + list(dff["Ano"])
    for scn in ["Baixa", "Base", "Alta"]:
        y_patr = [round(patr0_bi, 3)] + list((dff_scn[scn]["patr_fim"] / 1e9).round(3))
        figB.add_trace(go.Scatter(
            x=x_patr, y=y_patr, mode="lines+markers", yaxis="y2",
            name=f"Patrimônio — {scn}",
            line=dict(color=CENARIO_CORES[scn],
                      width=3 if scn == "Base" else 2,
                      dash="solid" if scn == "Base" else "dot")))

    figB.update_layout(
        barmode="stack", height=430,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Fluxos anuais (R$ bi)", rangemode="tozero"),
        yaxis2=dict(title="Patrimônio do fundo (R$ bi)", color="#1f77b4",
                    overlaying="y", side="right", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(figB, width='stretch')

    # ----- Tabelas --------------------------------------------------------- #
    tabela = pd.DataFrame({
        "Ano": dff["Ano"],
        "Selic (%)": (dff["Selic"] * 100).round(2),
        "Patrimônio início (R$ bi)": (dff["patr_inicio"] / 1e9).round(3),
        "Rendimento bruto (R$ bi)": (dff["rendimento_bruto"] / 1e9).round(3),
        "Saque (R$ bi)": (dff["saque"] / 1e9).round(3),
        "Aporte (R$ bi)": (dff["aporte"] / 1e9).round(3),
        "Patrimônio fim (R$ bi)": (dff["patr_fim"] / 1e9).round(3),
    })
    st.markdown("**Detalhamento — cenário base**")
    st.dataframe(tabela.set_index("Ano"), width='stretch')

    st.markdown("**Patrimônio ao fim do ano por cenário (R$ bi)**")
    cmp_f = pd.DataFrame(
        {scn: (dff_scn[scn]["patr_fim"] / 1e9).round(3).values for scn in ["Baixa", "Base", "Alta"]},
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
- Aporte (**LC 200/2022**) = **{APORTE_PCT_INCREMENTO:.0%} do incremento anual positivo**
  da arrecadação com Participação Especial + royalties excedentes (>5%), por cenário —
  **não** um percentual da receita cheia. A base é calibrada ao realizado do Tesouro.
- Patrimônio ao fim = patrimônio no início + reinvestido + aporte

O **primeiro gráfico** não usa valores em R$: apresenta o nível/sinal e a inclinação da
curva de rendimentos (Selic anual e índice acumulado base 100). O **segundo gráfico**
mostra a capitalização do fundo nos três cenários (Baixa/Base/Alta), com os fluxos
anuais do cenário base nas barras.

*Aportes pontuais de TAC e leilões (50%, LC 200/2022) não estão projetados.
Modelo simplificado: não considera marcação a mercado, tributação ou inflação.
Os cenários de receita vêm da aba Petróleo e Gás (bandas por volatilidade implícita).*
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
    cambio_modo = st.radio("Câmbio USD→BRL", ["Automático (Yahoo, diário)", "Manual"], horizontal=True)
    cambio_manual = st.number_input("USD/BRL manual", value=5.40, min_value=1.0,
                                    max_value=15.0, step=0.05,
                                    disabled=(cambio_modo != "Manual"))
    reajuste_oleo = st.slider(
        "Reajuste adicional do preço do óleo (% a.a.)", -20.0, 20.0, 0.0, 0.5,
        help="Sobreposição opcional à curva a termo (0 = usa a curva pura).") / 100.0

    st.subheader("Cenários de preço")
    conf_nivel = st.selectbox(
        "Amplitude das bandas (nível de confiança)", list(CONF_NIVEIS.keys()),
        help="Base = curva a termo. Cenários Baixa/Alta dimensionados pela volatilidade "
             "implícita (OVX) e pela amplitude escolhida.")

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
    conf_nivel=conf_nivel,
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
