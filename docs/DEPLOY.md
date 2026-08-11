# Nasazení na server

Produkční stack: **Postgres/TimescaleDB + FastAPI + dashboard za nginxem**,
před tím Traefik s Authentik forward-auth. Vše v `docker-compose.prod.yml`.

```
prohlížeč ──▶ Traefik ──▶ Authentik (forward auth) ──▶ web (nginx)
                                                        ├─ /            dashboard (statické soubory)
                                                        └─ /api/  ──▶  api (FastAPI) ──▶ db
```

Ven vede jediný router — na `web`. API nemá vlastní Traefik labely, takže
se k němu zvenčí nedá dostat jinak než přes stejný přihlášený origin.

---

## 1. Předpoklady na serveru

- Docker + Docker Compose
- Běžící **Traefik** v externí docker síti (výchozí jméno `proxy`)
- Běžící **Authentik** s forward-auth middlewarem pro Traefik
- DNS A záznam pro doménu dashboardu

## 2. Konfigurace

```bash
git clone <repo> garmin && cd garmin
cp .env.example .env
```

V `.env` doplň:

| Proměnná | Co to je |
|---|---|
| `POSTGRES_PASSWORD` | **Povinné.** Compose bez něj odmítne nastartovat. |
| `GARMIN_EMAIL`, `GARMIN_PASSWORD`, `GARMIN_DISPLAY_NAME` | Přihlášení k Garmin Connect pro noční sync |
| `DASHBOARD_HOST` | Doména, např. `dashboard.example.com` |
| `TRAEFIK_NETWORK` | Jméno externí sítě Traefiku (`proxy`) |
| `TRAEFIK_ENTRYPOINT`, `TRAEFIK_CERT_RESOLVER` | Podle tvé Traefik konfigurace |
| `AUTHENTIK_MIDDLEWARE` | Jméno middleware, např. `authentik@file` |
| `SYNC_CRON_HOUR` | Hodina nočního syncu (4 = 4:00 Europe/Prague), prázdné = vypnuto |
| `EXCLUDED_LOCATIONS` | Vyloučená GPS místa (domov, práce) |

## 3. Authentik

1. **Providers → Create → Proxy Provider**
   - Mode: *Forward auth (single application)*
   - External host: `https://<DASHBOARD_HOST>`
2. **Applications → Create** – přiřaď provider, nastav slug a přístupovou policy.
3. **Outposts** – přidej aplikaci do embedded outpostu.
4. V Traefiku musí existovat middleware, který posílá požadavky na
   `/outpost.goauthentik.io/auth/traefik`. Jeho jméno patří do
   `AUTHENTIK_MIDDLEWARE`.

Dashboard sám žádné přihlášení nemá — kdo projde Authentikem, vidí data.

## 4. První spuštění

```bash
docker compose -f docker-compose.prod.yml up -d --build db
docker compose -f docker-compose.prod.yml run --rm api alembic upgrade head
docker compose -f docker-compose.prod.yml up -d
```

Kontrola: `docker compose -f docker-compose.prod.yml ps` — všechny služby
`healthy`, a `https://<DASHBOARD_HOST>` po přihlášení ukáže dashboard.

## 5. Přenos historie

Prázdná databáze znamená prázdný dashboard. Znovustahovat vše z Garminu
nemá smysl (hodiny běhu, riziko rate limitu) — přenes dump:

```bash
# lokálně
docker exec garmin_db pg_dump -U garmin -Fc garmin > garmin.dump
scp garmin.dump server:/tmp/

# na serveru
docker cp /tmp/garmin.dump garmin_db:/tmp/
docker exec garmin_db pg_restore -U garmin -d garmin --clean --if-exists /tmp/garmin.dump
```

FIT soubory (`data/fit/`) přenes zvlášť do volume `garmin_data`, jinak by
`load --force` neměl z čeho parsovat:

```bash
rsync -av data/fit/ server:/tmp/fit/
docker cp /tmp/fit/. garmin_api:/app/data/fit/
```

Tokeny Garminu (`.garminconnect/`, `.garth/`) taky — sync se přihlašuje
**výhradně** uloženými tokeny, nové heslo si nevynutí:

```bash
docker cp .garminconnect/. garmin_api:/app/.garminconnect/
docker cp .garth/. garmin_api:/app/.garth/
docker compose -f docker-compose.prod.yml restart api
```

## 6. Provoz

```bash
# ruční sync
docker compose -f docker-compose.prod.yml exec api python scripts/main.py

# jen přepočet metrik
docker compose -f docker-compose.prod.yml exec api python scripts/main.py analyze

# co je v databázi
docker compose -f docker-compose.prod.yml exec api python scripts/main.py status

# logy
docker compose -f docker-compose.prod.yml logs -f api
```

Noční sync běží sám uvnitř `api` (APScheduler, `lifespan` v
`src/api/app.py`) — žádný cron na hostiteli.

### Aktualizace

```bash
git pull
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml run --rm api alembic upgrade head
```

### Záloha

```bash
docker exec garmin_db pg_dump -U garmin -Fc garmin > backup-$(date +%F).dump
```

Volume `garmin_data` (FIT soubory) je zdroj pravdy — databáze se z něj dá
postavit znovu, opačně ne.

---

## Řešení potíží

| Příznak | Příčina |
|---|---|
| `502` z Traefiku | `web` běží, ale `api` ne — `docker compose logs api` |
| Dashboard hlásí „Data se nepodařilo načíst" | API vrací chybu; zkus `docker compose exec web wget -qO- http://api:8000/health` |
| Prázdný dashboard, hlášení o chybějících metrikách | Databáze je bez dat — viz krok 5 |
| Sync padá na autentizaci | Nepřenesené nebo propadlé tokeny; obnov je lokálně přes `scripts/setup/seed_token.py` a znovu zkopíruj |
| Nekonečné přesměrování na login | Authentik provider není v outpostu, nebo se neshoduje External host s doménou |
