# Intradagsarkiv

5-minutersstaplar för alla instrument på sajten, en krypterad fil per handelsdag
(`days/ÅÅÅÅ/MM/ÅÅÅÅ-MM-DD.enc.json`). Skrivs av `pipeline/intraday_archive.py` två gånger per
handelsdag. Nyckeln finns i `auth.json`, inlindad per konto i SITE_USERS. Ändra inte grenen för hand.
