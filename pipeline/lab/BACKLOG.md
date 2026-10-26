# Strategilabbets backlog

Den här listan styr strategiagenten (en schemalagd Claude-session varje vecka). Agenten tar
**en** punkt åt gången uppifrån, implementerar den i `pipeline/strategy_lab.py`, skriver tester
och öppnar en pull request. Ägaren granskar och slår ihop. Stryk punkter med `[x]` när de är klara
och lägg gärna till nya idéer längst ned.

## Regler för agenten

- Lägg till strategier som en ny post i `FAMILIES`. Per-instrument-strategier: `fn(df, **params)`
  som returnerar position 0/1 per dag, bestämd vid stängning. Rotationsstrategier: `portfolio=True`
  och `fn(prices, **params)` som returnerar en DataFrame med positioner.
- **Ingen framåtblick.** En signal får bara använda data fram till och med dagens stängning.
  Lägg till ett test som visar det när strategin använder rullande fönster eller `shift`.
- **Små parameterrutnät: högst 4 varianter per strategi.** Varje extra variant ökar risken att
  något ser bra ut av slump. Välj etablerade standardvärden från litteraturen, inte finjusterade tal.
- Endast köp/stå utanför om inte punkten säger annat. Blankande strategier markeras `short=True`,
  returnerar −1 för kort position och betalar `SHORT_COST_PCT_YEAR`.
- Ändra inte kostnadsmodellen, utvärderingen (`run`, `walk_forward`, `metrics`) eller de
  live-regler som sajten visar (`mockup/analysis.js`) utan att punkten uttryckligen säger det.
- Alla tester ska gå igenom: `python -m unittest discover -s tests -t .`
- PR-beskrivningen ska säga vilken källa strategin kommer från och vilka parametrar som valts och varför.
  Den kan inte innehålla resultat — labbet körs mot riktig data först när PR:en slagits ihop.

## Att göra

Ordningen är satt av ägaren (2026-10-12) efter fyra veckors resultat: tajmingregler på enskilda aktier
ger ungefär samma avkastning per risk som köp och behåll, så fokus flyttas till spridning mellan
tillgångsslag, faktorer och väl studerade säsongsmönster.

- [ ] **Momentum 12–1** — rotation efter avkastningen från 12 månader till 1 månad sedan (senaste månaden
      hoppas över), topp 5 eller 10, månadsvis. Källa: Jegadeesh & Titman 1993.
- [ ] **Marknadsregimfilter** — handla bara aktier när deras index (OMXS30 för Norden, SPX för USA)
      ligger över sitt 200-dagars medelvärde. Kombinera med Donchian och RSI(2).
- [ ] **Säsongsmönster** — (a) månadsskiftet: äg index de sista 2 och första 3 handelsdagarna i månaden
      (Ariel 1987; Lakonishok & Smidt 1988); (b) Sell in May: äg index november–april, kontant maj–oktober
      (Bouman & Jacobsen 2002). Testas på index och på veckodatan (10 år).
- [ ] **IBS-rekyl (Internal Bar Strength)** — köp när (stängning − lägsta) / (högsta − lägsta) < 0,2 i upptrend,
      sälj när IBS > 0,8.

Strukna 2026-10-12: Keltner-utbrott, Supertrend och Ichimoku. De är varianter av samma trendidé som
Donchian och tidsseriemomentum (redan testade) och skulle främst öka risken att något ser bra ut av slump.

## Klart

- [x] Köp och behåll, Faber, medelvärdeskorsning, Donchian, tidsseriemomentum, MACD, RSI(2),
      Bollinger, 52-veckorshögsta, ATR-stop, relativ styrka (rotation).
- [x] Köp/blanka-versioner: pris mot medelvärde, medelvärdeskorsning, tidsseriemomentum,
      Donchian och RSI(2), med årlig kostnad för korta positioner (`SHORT_COST_PCT_YEAR`).
- [x] Veckodata — labbet körs även på veckoserien (10 års historik) med de långsiktiga strategierna
      (`WEEKLY_FAMILIES`, inställningar i veckor).
- [x] Kombination av strategier — likaviktad portfölj av de 3 strategier som haft bäst Sharpe de senaste
      två åren, omvald lika ofta som mästaren (`walk_forward_combo`), jämförs mot mästaren och köp och behåll.
- [x] Volatilitetsstyrd exponering — Faber, Donchian, ATR-stop och tidsseriemomentum där hela portföljen
      skalas mot 15 % årlig volatilitet (max 1,5 gånger, lånekostnad `LEVERAGE_COST_PCT_YEAR`).
- [x] Dubbelt momentum (Antonacci) — rotation mellan OMXS30, S&P 500, Nasdaq 100 och guld, 6 och 12 månader.
- [x] Lista utan efterhandsval — nuvarande och tidigare OMXS30-bolag (`config/lab_omxs30.csv`), tio års
      dagsdata med egen hämtning, labbet körs på listan (`omxs30` i lab.json). Avnoterade bolag utan kurser redovisas.
- [x] Faber GTAA — fem tillgångsslag (SPY, EFA, IEF, VNQ, DBC, `config/lab_gtaa.csv`), 20 års utdelningsjusterad
      dagsdata, signal vid månadsskiften mot 10- och 12-månaders medelvärdet (`gtaa` i lab.json).
- [x] Låg volatilitet — rotation till den lugnaste femtedelen av aktierna vid varje månadsskifte
      (63 eller 252 dagar; 13 eller 52 veckor), körs även på OMXS30-listan utan efterhandsval.
