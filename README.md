# Dataset Explorer

Dataset Explorer is a local Streamlit app for exploring CSV files. It shows the file overview, missing and duplicate counts, a preview, and category totals in a bar chart. Uploaded data is not automatically cleaned or changed.

The app uses a teal background (`#137166`), bright green accents (`#22ff05`), and two laughing cat caricatures in the welcome banner.

## Setup and restart on this computer

Open PowerShell and run these commands to install the listed project dependencies into the existing virtual environment:

```powershell
Set-Location -LiteralPath 'C:\Users\Student\Desktop\Dataset explorer'
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

To start or restart the app, run:

```powershell
Set-Location -LiteralPath 'C:\Users\Student\Desktop\Dataset explorer'
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

Open [http://localhost:8502/](http://localhost:8502/). Keep the PowerShell window open while using the app; press `Ctrl+C` there to stop it.

## Environment checked

The project interpreter and imports were checked on this computer: Python `3.12.10`, pandas `3.1.0`, and Streamlit `1.66.1`. Streamlit served the app at port `8502` with an HTTP 200 response. Automated CSV checks and browser-manual status are recorded in `checks.txt`.

## Use it

Upload a UTF-8 CSV file with a header, choose comma or semicolon as the separator, and keep the file at or below 5 MB. The app shows row and column counts, column types, missing-value counts and percentages, exact duplicate rows after their first occurrence, and a preview of up to 100 rows.

Missing counts use pandas `read_csv` defaults: empty fields and recognized default NA markers count as missing. Space-only text and zero values do not. Blank lines are skipped. The app does not clean or change uploaded values.

For the bar chart, choose a category column and a numeric column. It sums non-missing numbers for each non-missing category, shows up to 20 categories in descending total order, and lists all category totals in the table below. A category with only missing numbers is labeled `No data`; rows with a missing category are counted separately.
