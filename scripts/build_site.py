import json
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs


ROOT = Path(__file__).resolve().parents[1]
RISK_PATH = ROOT / "artifacts" / "submissions" / "risk_scores__xgboost__tuned_final.csv"
SITE_DIR = ROOT / "site"
OUT_PATH = SITE_DIR / "index.html"
DOCS_DIR = ROOT / "docs"
DOCS_PATH = DOCS_DIR / "index.html"


COUNTRY_KO = {
    "AFG": "아프가니스탄",
    "ARM": "아르메니아",
    "AZE": "아제르바이잔",
    "BFA": "부르키나파소",
    "BGD": "방글라데시",
    "CAF": "중앙아프리카공화국",
    "CIV": "코트디부아르",
    "CMR": "카메룬",
    "COD": "콩고민주공화국",
    "COL": "콜롬비아",
    "DZA": "알제리",
    "ECU": "에콰도르",
    "EGY": "이집트",
    "ERI": "에리트레아",
    "ETH": "에티오피아",
    "GIN": "기니",
    "GNB": "기니비사우",
    "GTM": "과테말라",
    "HND": "온두라스",
    "HTI": "아이티",
    "IDN": "인도네시아",
    "IND": "인도",
    "IRN": "이란",
    "IRQ": "이라크",
    "ISR": "이스라엘",
    "KEN": "케냐",
    "KGZ": "키르기스스탄",
    "LBN": "레바논",
    "LBY": "리비아",
    "MDG": "마다가스카르",
    "MEX": "멕시코",
    "MLI": "말리",
    "MMR": "미얀마",
    "MOZ": "모잠비크",
    "NER": "니제르",
    "NGA": "나이지리아",
    "PAK": "파키스탄",
    "PHL": "필리핀",
    "PSE": "팔레스타인",
    "RUS": "러시아",
    "SAU": "사우디아라비아",
    "SDN": "수단",
    "SEN": "세네갈",
    "SLE": "시에라리온",
    "SOM": "소말리아",
    "SSD": "남수단",
    "SYR": "시리아",
    "TCD": "차드",
    "TGO": "토고",
    "THA": "태국",
    "TJK": "타지키스탄",
    "TUN": "튀니지",
    "TUR": "튀르키예",
    "UGA": "우간다",
    "UKR": "우크라이나",
    "VEN": "베네수엘라",
    "YEM": "예멘",
    "ZWE": "짐바브웨",
}


def risk_label(score):
    if score >= 80:
        return "매우 높음"
    if score >= 65:
        return "높음"
    if score >= 45:
        return "경계"
    if score >= 25:
        return "주의"
    return "낮음"


def build_payload():
    df = pd.read_csv(RISK_PATH)
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].copy()
    all_scores = df.copy()
    latest["country_name_ko"] = latest["country"].map(COUNTRY_KO).fillna(latest["country"])
    latest["risk_label"] = latest["risk_score"].map(risk_label)
    latest = latest.sort_values("risk_score", ascending=False)
    all_scores["country_name_ko"] = all_scores["country"].map(COUNTRY_KO).fillna(all_scores["country"])
    all_scores["risk_label"] = all_scores["risk_score"].map(risk_label)
    all_scores = all_scores.sort_values(["date", "risk_score"], ascending=[True, False])

    trend = df.sort_values(["country", "date"]).copy()
    trend["country_name_ko"] = trend["country"].map(COUNTRY_KO).fillna(trend["country"])

    return {
        "dates": sorted(df["date"].unique().tolist()),
        "latest_date": latest_date,
        "latest": latest.to_dict(orient="records"),
        "records": all_scores.to_dict(orient="records"),
        "trend": {
            country: group[
                [
                    "date",
                    "risk_score",
                    "y_prob",
                    "B_score",
                    "C_state",
                    "F_score",
                    "U_score",
                    "C_score",
                    "S_score",
                    "I_score",
                ]
            ].to_dict(orient="records")
            for country, group in trend.groupby("country", sort=False)
        },
    }


def main():
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    plotly_js = get_plotlyjs()
    html = HTML_TEMPLATE.replace("__PLOTLY_JS__", plotly_js).replace(
        "__APP_DATA__", json.dumps(payload, ensure_ascii=False)
    )
    OUT_PATH.write_text(html, encoding="utf-8")
    DOCS_PATH.write_text(html, encoding="utf-8")
    print(f"saved: {OUT_PATH}")
    print(f"saved for GitHub Pages: {DOCS_PATH}")
    print(f"latest_date: {payload['latest_date']}")
    print(f"countries: {len(payload['latest'])}")
    print("open directly: site/index.html")
    print("or serve with: python -m http.server 8000 -d site")
    print("then open: http://localhost:8000")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Global Risk Monitor Korea</title>
  <style>
    :root {
      --bg: #f6f7f2;
      --surface: #ffffff;
      --ink: #18201c;
      --muted: #64706a;
      --line: #dce1d7;
      --green: #1b9e77;
      --yellow: #e9c46a;
      --orange: #f08a4b;
      --red: #d94f45;
      --deep: #7f1d1d;
      --blue: #2f6f9f;
      --shadow: 0 18px 50px rgba(28, 38, 32, 0.12);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "Pretendard", "Noto Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
      background:
        linear-gradient(135deg, rgba(47, 111, 159, 0.12), transparent 34%),
        linear-gradient(315deg, rgba(233, 196, 106, 0.16), transparent 28%),
        var(--bg);
    }

    .app {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      grid-template-rows: auto minmax(0, 1fr);
      min-height: 100vh;
      gap: 18px;
      padding: 22px;
    }

    header {
      grid-column: 1 / -1;
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 18px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 16px;
    }

    h1 {
      margin: 0;
      font-size: clamp(26px, 4vw, 46px);
      line-height: 1.05;
      letter-spacing: 0;
    }

    .subtitle {
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.6;
      max-width: 820px;
    }

    .date-pill {
      flex: none;
      padding: 10px 14px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.72);
      border-radius: 8px;
      color: var(--muted);
      font-size: 14px;
    }

    .date-pill label {
      display: block;
      margin-bottom: 6px;
      font-size: 12px;
      font-weight: 700;
    }

    .date-pill select {
      width: 150px;
      border: 0;
      background: transparent;
      color: var(--ink);
      font: inherit;
      font-weight: 800;
      outline: none;
      cursor: pointer;
    }

    .map-area {
      min-height: 680px;
      background: rgba(255, 255, 255, 0.76);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
      position: relative;
    }

    #map {
      width: 100%;
      height: 100%;
      min-height: 680px;
    }

    aside {
      display: flex;
      flex-direction: column;
      gap: 14px;
      min-width: 0;
    }

    .panel {
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 18px;
    }

    .country-title {
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 12px;
    }

    .country-title h2 {
      margin: 0;
      font-size: 25px;
      line-height: 1.18;
      letter-spacing: 0;
    }

    .label {
      border-radius: 999px;
      padding: 6px 10px;
      color: white;
      font-size: 13px;
      font-weight: 700;
      white-space: nowrap;
    }

    .score {
      display: flex;
      align-items: baseline;
      gap: 8px;
      margin-top: 18px;
    }

    .score strong {
      font-size: 52px;
      line-height: 1;
      letter-spacing: 0;
    }

    .score span {
      color: var(--muted);
      font-weight: 700;
    }

    .bar-list {
      display: grid;
      gap: 12px;
      margin-top: 18px;
    }

    .metric-row {
      display: grid;
      grid-template-columns: 72px 1fr 48px;
      gap: 10px;
      align-items: center;
      font-size: 13px;
      color: var(--muted);
    }

    .track {
      height: 9px;
      border-radius: 999px;
      background: #e7ebe4;
      overflow: hidden;
    }

    .fill {
      height: 100%;
      border-radius: inherit;
      background: var(--blue);
    }

    .explain {
      margin: 16px 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.65;
    }

    .ranking {
      max-height: 420px;
      overflow: auto;
      padding: 0;
    }

    .ranking h3 {
      margin: 18px 18px 10px;
      font-size: 16px;
    }

    .rank-item {
      width: 100%;
      border: 0;
      border-top: 1px solid var(--line);
      background: transparent;
      display: grid;
      grid-template-columns: 28px 1fr 54px;
      gap: 10px;
      align-items: center;
      padding: 12px 18px;
      color: var(--ink);
      text-align: left;
      cursor: pointer;
      font-family: inherit;
      font-size: 14px;
    }

    .rank-item:hover {
      background: rgba(47, 111, 159, 0.08);
    }

    .rank-score {
      font-weight: 800;
      text-align: right;
    }

    #trend {
      width: 100%;
      height: 210px;
    }

    @media (max-width: 980px) {
      .app {
        grid-template-columns: 1fr;
        padding: 14px;
      }

      header {
        align-items: start;
        flex-direction: column;
      }

      .map-area,
      #map {
        min-height: 520px;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div>
        <h1>세계 분쟁 위험도 모니터</h1>
        <p class="subtitle">뉴스를 깊게 따라가지 않아도 국가별 위험 신호를 빠르게 볼 수 있도록 만든 한국어 대시보드입니다. 지도에서 국가를 선택하면 예측 위험도와 그 점수를 구성하는 요소를 함께 확인할 수 있습니다.</p>
      </div>
      <div class="date-pill">
        <label for="dateSelect">기준일</label>
        <select id="dateSelect" aria-label="기준일 선택"></select>
      </div>
    </header>

    <main class="map-area">
      <div id="map"></div>
    </main>

    <aside>
      <section class="panel" id="detailPanel">
        <div class="country-title">
          <h2 id="countryName">국가 선택</h2>
          <span class="label" id="riskLabel">-</span>
        </div>
        <div class="score">
          <strong id="riskScore">--</strong>
          <span>/ 100</span>
        </div>
        <div class="bar-list" id="metricBars"></div>
        <p class="explain" id="explainText">지도에서 색이 진한 국가는 최근 사건, 뉴스 신호, 장기 위험 prior, 모델 예측값이 함께 높게 나타난 곳입니다.</p>
      </section>

      <section class="panel">
        <div id="trend"></div>
      </section>

      <section class="panel ranking">
        <h3>위험도 상위 국가</h3>
        <div id="ranking"></div>
      </section>
    </aside>
  </div>

  <script>__PLOTLY_JS__</script>
  <script>
    const APP_DATA = __APP_DATA__;
    const recordsByDate = APP_DATA.records.reduce((acc, row) => {
      if (!acc[row.date]) acc[row.date] = [];
      acc[row.date].push(row);
      return acc;
    }, {});
    let selectedDate = APP_DATA.latest_date;
    let latest = recordsByDate[selectedDate] || APP_DATA.latest;
    let byCountry = Object.fromEntries(latest.map(d => [d.country, d]));
    let selectedCountry = latest[0].country;
    const colors = {
      "낮음": "#1b9e77",
      "주의": "#e9c46a",
      "경계": "#f08a4b",
      "높음": "#d94f45",
      "매우 높음": "#7f1d1d"
    };

    const dateSelect = document.getElementById("dateSelect");
    dateSelect.innerHTML = APP_DATA.dates.map(date => `<option value="${date}">${date}</option>`).join("");
    dateSelect.value = selectedDate;
    dateSelect.addEventListener("change", () => {
      selectedDate = dateSelect.value;
      latest = recordsByDate[selectedDate] || [];
      byCountry = Object.fromEntries(latest.map(d => [d.country, d]));
      if (!byCountry[selectedCountry]) selectedCountry = latest[0]?.country;
      updateMap();
      drawRanking();
      selectCountry(selectedCountry);
    });

    function riskColor(score) {
      if (score >= 80) return colors["매우 높음"];
      if (score >= 65) return colors["높음"];
      if (score >= 45) return colors["경계"];
      if (score >= 25) return colors["주의"];
      return colors["낮음"];
    }

    function fmt(value) {
      return Number(value).toFixed(1);
    }

    function drawMap() {
      const trace = {
        type: "choropleth",
        locationmode: "ISO-3",
        locations: latest.map(d => d.country),
        z: latest.map(d => d.risk_score),
        text: latest.map(d => `${d.country_name_ko} (${d.country})`),
        customdata: latest.map(d => d.country),
        hovertemplate: "<b>%{text}</b><br>위험도 %{z:.1f}/100<extra></extra>",
        zmin: 0,
        zmax: 100,
        colorscale: [
          [0.00, "#1b9e77"],
          [0.25, "#e9c46a"],
          [0.45, "#f08a4b"],
          [0.65, "#d94f45"],
          [1.00, "#7f1d1d"]
        ],
        marker: { line: { color: "rgba(255,255,255,0.78)", width: 0.6 } },
        colorbar: {
          title: "위험도",
          thickness: 14,
          len: 0.72
        }
      };

      const layout = {
        margin: { l: 0, r: 0, t: 0, b: 0 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        geo: {
          projection: { type: "natural earth" },
          bgcolor: "rgba(0,0,0,0)",
          showframe: false,
          showcoastlines: true,
          coastlinecolor: "#aeb8af",
          showcountries: true,
          countrycolor: "#ffffff",
          showland: true,
          landcolor: "#eef1ea",
          showocean: true,
          oceancolor: "#dce8ee",
          lataxis: { range: [-58, 84] }
        }
      };

      Plotly.newPlot("map", [trace], layout, { responsive: true, displayModeBar: false });
      document.getElementById("map").on("plotly_click", ev => {
        const iso = ev.points[0].customdata;
        selectCountry(iso);
      });
    }

    function updateMap() {
      const update = {
        locations: [latest.map(d => d.country)],
        z: [latest.map(d => d.risk_score)],
        text: [latest.map(d => `${d.country_name_ko} (${d.country})`)],
        customdata: [latest.map(d => d.country)]
      };
      Plotly.restyle("map", update);
    }

    function drawTrend(iso) {
      const rows = APP_DATA.trend[iso] || [];
      const country = byCountry[iso];
      const trace = {
        type: "scatter",
        mode: "lines",
        x: rows.map(d => d.date),
        y: rows.map(d => d.risk_score),
        line: { color: riskColor(country.risk_score), width: 3 },
        hovertemplate: "%{x}<br>위험도 %{y:.1f}<extra></extra>"
      };
      const layout = {
        title: { text: `${country.country_name_ko} 최근 위험도 흐름`, font: { size: 15 } },
        margin: { l: 36, r: 8, t: 42, b: 28 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        yaxis: { range: [0, 100], gridcolor: "#e7ebe4" },
        xaxis: { showgrid: false },
        shapes: [{
          type: "line",
          x0: selectedDate,
          x1: selectedDate,
          y0: 0,
          y1: 100,
          yref: "y",
          line: { color: "#18201c", width: 1, dash: "dot" }
        }]
      };
      Plotly.newPlot("trend", [trace], layout, { responsive: true, displayModeBar: false });
    }

    function metricBar(label, value, color) {
      return `
        <div class="metric-row">
          <span>${label}</span>
          <div class="track"><div class="fill" style="width:${Math.max(0, Math.min(100, value))}%; background:${color};"></div></div>
          <strong>${fmt(value)}</strong>
        </div>
      `;
    }

    function selectCountry(iso) {
      const d = byCountry[iso];
      if (!d) return;
      selectedCountry = iso;

      const label = document.getElementById("riskLabel");
      label.textContent = d.risk_label;
      label.style.background = riskColor(d.risk_score);

      document.getElementById("countryName").textContent = `${d.country_name_ko} (${d.country})`;
      document.getElementById("riskScore").textContent = fmt(d.risk_score);
      document.getElementById("metricBars").innerHTML = [
        metricBar("장기위험", d.B_score, "#2f6f9f"),
        metricBar("현재상태", d.C_state, "#f08a4b"),
        metricBar("모델예측", d.F_score, "#d94f45"),
        metricBar("민간피해", d.U_score, "#7f1d1d"),
        metricBar("분쟁신호", d.C_score, "#b45309"),
        metricBar("뉴스긴장", d.S_score, "#84631c"),
        metricBar("정보량", d.I_score, "#316b83")
      ].join("");

      document.getElementById("explainText").textContent =
        `${selectedDate} 기준 단기 악화 가능성은 ${(d.y_prob * 100).toFixed(2)}%로 추정됩니다. 종합 점수는 장기 위험도, 최근 상태 변화, 모델 예측값을 합산해 0-100점으로 환산했습니다.`;

      drawTrend(iso);
    }

    function drawRanking() {
      const box = document.getElementById("ranking");
      box.innerHTML = latest.slice(0, 16).map((d, i) => `
        <button class="rank-item" data-country="${d.country}">
          <span>${i + 1}</span>
          <span>${d.country_name_ko} (${d.country})</span>
          <span class="rank-score">${fmt(d.risk_score)}</span>
        </button>
      `).join("");
      box.querySelectorAll("button").forEach(btn => {
        btn.addEventListener("click", () => selectCountry(btn.dataset.country));
      });
    }

    drawMap();
    drawRanking();
    selectCountry(latest[0].country);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
