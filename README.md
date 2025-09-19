# SmartInventory (StockPulse)

[(Click here for Live Demo)](https://smartinventorymachinelearning.streamlit.app/)

A Streamlit-based retail forecasting and inventory optimization app. Upload sales data, explore trends, generate demand forecasts, and get inventory recommendations.

## Features
- Data upload, validation, and cleaning
- Interactive exploration with Plotly charts
- Forecasting engine and performance tracking
- Inventory optimization (EOQ, safety stock, reorder point)
- Persistent storage via MySQL (configurable); in-memory fallback when DB is unavailable

## Project Structure
- `app.py` — Streamlit multi-page app entry
- `pages/` — Streamlit pages (Data Upload, Exploration, Forecasting, Inventory, Metrics)
- `utils/`
  - `data_processing.py` — Validation, transforms, aggregations
  - `forecasting.py` — Forecast utilities
  - `inventory_optimization.py` — Inventory logic
  - `database.py` — Database manager (MySQL via SQLAlchemy) used by the app
- Top-level `database.py` — Lightweight DB helper (MySQL)
- `.streamlit/config.toml` — Streamlit server config (host/port/theme)
- `pyproject.toml` — Dependencies

## Requirements
- Python 3.11+
- MySQL 8.x server (or compatible)
- Windows PowerShell (examples use PowerShell syntax)

## Recent Changes
- Database backend migrated from PostgreSQL (psycopg2) to MySQL (PyMySQL + SQLAlchemy)
- Env vars now use DATABASE_URL or DB_*/MYSQL_* variants
- DDL updated for MySQL (AUTO_INCREMENT, per-table TRUNCATE, CREATE INDEX behavior)
- Streamlit server binds to 127.0.0.1:8501 by default via `.streamlit/config.toml`

## Setup (Windows)
1) Create and activate a virtual environment:
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2) Install dependencies:
```powershell
pip install -U pip
pip install numpy pandas plotly pymysql scikit-learn scipy sqlalchemy streamlit
```

3) Create a MySQL database (one-time):
- Start MySQL and create a database, e.g. `smartinventory`.
```sql
-- Example SQL you can run in MySQL
CREATE DATABASE IF NOT EXISTS smartinventory CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
-- Optionally create a dedicated user
CREATE USER IF NOT EXISTS 'smartuser'@'localhost' IDENTIFIED BY 'strong_password';
GRANT ALL PRIVILEGES ON smartinventory.* TO 'smartuser'@'localhost';
FLUSH PRIVILEGES;
```

4) Configure database connection (optional):
- The app reads a full `DATABASE_URL` or individual vars. For MySQL use one of:
```powershell
# Option A: single URL
$env:DATABASE_URL = "mysql+pymysql://root:password@127.0.0.1:3306/smartinventory"

# Option B: individual variables (DB_* or MYSQL_*)
$env:DB_HOST = "127.0.0.1"      # or MYSQL_HOST
$env:DB_PORT = "3306"           # or MYSQL_PORT
$env:DB_NAME = "smartinventory" # or MYSQL_DB
$env:DB_USER = "root"           # or MYSQL_USER
$env:DB_PASSWORD = "password"   # or MYSQL_PASSWORD
```
Defaults if not set: host 127.0.0.1, port 3306, db smartinventory, user root, empty password.

Or use a `.env` file (preferred for local dev):
1. Copy `.env.example` to `.env`
2. Edit values inside `.env` (DATABASE_URL or DB_* / MYSQL_* vars)
3. The app automatically loads `.env` on start


Env var precedence:
1. If `DATABASE_URL` is set, it is used directly
2. Otherwise individual vars (DB_* or MYSQL_*) are used to build the URL

Tip: If your password contains special characters, URL-encode it in `DATABASE_URL` (e.g., `p@ss` -> `p%40ss`).

5) Server host/port (optional):
- Controlled by `.streamlit/config.toml`. Current default:
```
[server]
headless = true
address = "127.0.0.1"
port = 8501
```
To expose on your LAN: set `address = "0.0.0.0"` and keep a preferred port.

## Run
```powershell
streamlit run app.py
```
Then open http://127.0.0.1:8501 in your browser.

## Data Format
Minimum required columns (can be mapped on upload):
- date (parseable)
- product_name (string)
- sales_quantity (numeric)
Optional: price, category, store_id. The app derives revenue and calendar fields.

## Database Notes (MySQL)
- Driver: PyMySQL via SQLAlchemy. URL: `mysql+pymysql://user:pass@host:port/dbname`.
- Tables are created automatically on first DB connection.
- Clearing data uses TRUNCATE per table.
- If DB connection fails, the app uses in-memory storage (per session).

## Troubleshooting
- Connection error: verify credentials, DB exists, privileges granted.
- Import error: ensure the venv is activated before running.
- Port in use: change `port` in `.streamlit/config.toml`.
- Access from other devices: set `address = "0.0.0.0"` and allow Windows Firewall.
- MySQL auth plugin errors: ensure your user uses `mysql_native_password` or a compatible plugin.
- SSL/timeout issues: add query params to `DATABASE_URL` as needed, e.g. `?charset=utf8mb4`.


