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

Ordningen är satt av ägaren (2026-09-23) efter genomgången av labbets första resultat.

- [ ] **Kombination av strategier** — i stället för en enda mästare: en likaviktad portfölj av de 2–3
      strategier som var robustast (bäst Sharpe i den tidiga perioden). Mästarvalet jagar i dag den strategi
      som gått bäst senast och byter för sent; en kombination ska jämföras mot både mästaren och köp och behåll.
- [ ] **Volatilitetsstyrd exponering** — överlägg på trendstrategierna (Trend med ATR-stop, Donchian, Faber):
      skala positionen så att den årliga volatiliteten blir ca 15 % (maxhävstång 1,5). Kräver att positioner
      får vara mellan 0 och 1,5.
- [ ] **Lista utan efterhandsval** — testa på ett urval som inte är valt för att det gått bra, t.ex. alla
      nuvarande och tidigare bolag i OMXS30. Kräver ny datahämtning och ett eget universum i labbet.
- [ ] **Marknadsregimfilter** — handla bara aktier när deras index (OMXS30 för Norden, SPX för USA)
      ligger över sitt 200-dagars medelvärde. Kombinera med Donchian och RSI(2).
- [ ] **Keltner-utbrott** — köp stängning över EMA(20) + 2 × ATR(10), sälj under EMA(20).
- [ ] **Supertrend** — ATR(10) × 3, standardinställning.
- [ ] **IBS-rekyl (Internal Bar Strength)** — köp när (stängning − lägsta) / (högsta − lägsta) < 0,2 i upptrend,
      sälj när IBS > 0,8.
- [ ] **Månadsskiftet (turn of the month)** — äg index de sista 2 och första 3 handelsdagarna i månaden.
- [ ] **Ichimoku** — pris över molnet och Tenkan över Kijun (9/26/52).

## Klart

- [x] Köp och behåll, Faber, medelvärdeskorsning, Donchian, tidsseriemomentum, MACD, RSI(2),
      Bollinger, 52-veckorshögsta, ATR-stop, relativ styrka (rotation).
- [x] Köp/blanka-versioner: pris mot medelvärde, medelvärdeskorsning, tidsseriemomentum,
      Donchian och RSI(2), med årlig kostnad för korta positioner (`SHORT_COST_PCT_YEAR`).
- [x] Veckodata — labbet körs även på veckoserien (10 års historik) med de långsiktiga strategierna
      (`WEEKLY_FAMILIES`, inställningar i veckor).
