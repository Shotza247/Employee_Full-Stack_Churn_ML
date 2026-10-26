# Implementation Plan: Declutter Tab 1 Ingestion Studio with Segmented Ingestion Mode Selector

## Project Status: ✅ COMPLETED & CLOSED (Production Ready)

## Top-Level Overview
Refactor **Tab 1: Data Ingestion, Synthesis & Lineage Studio** in [`streamlit_app.py`](streamlit_app.py:261) by replacing the cramped multi-column layout with a top-level **Segmented Mode Selector** (using `st.pills` or `st.radio` with horizontal layout). 
This will present one focused ingestion workflow at a time across full container width:
1. 📥 **Batch CSV Import & Auto-Repair** (Packaged Sample Roster Downloads, CSV File Uploader, Data Quality Assessment, Row Preview, Duplicate Guard, and Post-Import 1-Click Faker Auto-Repair)
2. 🎲 **Synthetic Batch Generator** (Faker Identity Producer with Churn Bias Slider and Batch Size controls)
3. ➕ **Single Employee Entry** (Streamlined 2-Column Manual Form with Instant Local ML Risk Scoring)

The bottom sections — **🔍 Data Lineage & Provenance Explorer** and **📊 SQLite Database Health & Schema Inspector** — will remain full-width, prominent, and completely untouched.

---

## Sub-Tasks

### Sub-Task 1: Implement Segmented Ingestion Mode Selector in Tab 1
- **Intent**: Replace the cramped two-column layout (`ingest_col1`, `ingest_col2`) with a clean, horizontal Segmented Mode Selector at the top of Tab 1.
- **Expected Outcomes**:
  - A responsive segmented control (`st.pills` or `st.radio(horizontal=True)`) displays 3 clean options:
    - 📥 **Batch CSV Import & Repair**
    - 🎲 **Synthetic Batch Generator**
    - 👤 **Single Employee Entry**
  - Only the active mode container is rendered with full width, eliminating visual clutter.
- **Todo List**:
  1. Add horizontal mode selector widget below the Tab 1 header.
  2. Structure conditional containers for each selected mode.
- **Relevant Context**:
  - `streamlit_app.py:261-275`
- **Status**: `[x] done`

---

### Sub-Task 2: Build Streamlined Batch CSV Import & Auto-Repair Mode
- **Intent**: Provide a spacious, dedicated container for CSV uploads, sample dataset downloads, data quality scorecards, table previews, duplicate file guards, and post-import auto-repair.
- **Expected Outcomes**:
  - Sample dataset downloader rendered in a clean collapsible expander.
  - Uploaded file quality assessment, column mapping, and preview table displayed with full container width.
  - Import button with duplicate filename ingestion blocker (`is_source_imported`).
  - Post-import resolution card displaying rejected row breakdowns and the 1-click Faker Auto-Repair button.
- **Todo List**:
  1. Migrate Sample CSV Downloader, Uploader, Quality Diagnosis, and Import actions into the Batch CSV container.
  2. Retain `st.session_state["last_import"]` persistence and post-import Faker Auto-Repair workflow.
  3. Ensure file-specific `source_system` tagging and duplicate guards remain active.
- **Relevant Context**:
  - `streamlit_app.py:276-369`
  - `churn_app/database.py:diagnose_dataframe_quality, is_source_imported, import_employee_dataframe`
  - `churn_app/faker_producer.py:repair_rejected_rows`
- **Status**: `[x] done`

---

### Sub-Task 3: Build Dedicated Synthetic Batch Generator Mode
- **Intent**: Provide a dedicated, spacious control center for Faker synthetic workforce generation with instant database ingestion and lineage tracking.
- **Expected Outcomes**:
  - Clean slider controls for Record Batch Size (5–100) and Churn Signal Bias (5%–80%).
  - 1-click batch generation with instant SQLite ingestion tagged with `source_system="faker_synthetic"`.
  - Lineage stats dynamically updated without UI crowding.
- **Todo List**:
  1. Implement the Synthetic Batch Producer inside its dedicated mode container.
  2. Include an informative preview of generated identity distributions.
- **Relevant Context**:
  - `streamlit_app.py:371-381`
  - `churn_app/faker_producer.py:generate_synthetic_batch`
- **Status**: `[x] done`

---

### Sub-Task 4: Build Dedicated Single Employee Entry & Instant Scoring Mode
- **Intent**: Provide an uncluttered manual ingestion form with immediate real-time ML risk inference and SQLite audit logging.
- **Expected Outcomes**:
  - Clean 2-column form for employee demographics, South African Rand compensation, department, and contract type.
  - On submission: record persisted to SQLite with `source_system="manual_entry"`, scored against `churn_model.pkl`, and risk badge displayed.
- **Todo List**:
  1. Implement the single employee form in its dedicated mode container.
  2. Render the post-submission inference risk scorecard and probability breakdown cleanly.
- **Relevant Context**:
  - `streamlit_app.py:382-429`
  - `churn_app/database.py:insert_employee, save_inference_predictions`
- **Status**: `[x] done`

---

### Sub-Task 5: Verify Data Lineage Explorer, Database Health Scorecards & E2E Validation
- **Intent**: Verify that all ingestion channels cleanly feed into the full-width Data Lineage Explorer and Database Health Inspector without errors.
- **Expected Outcomes**:
  - **🔍 Data Lineage & Provenance Explorer** dynamically reflects new records across all source systems (`<file_name>.csv`, `faker_repaired (<file_name>.csv)`, `faker_synthetic`, `manual_entry`, `seed_baseline`).
  - **📊 SQLite Database Health & Schema Inspector** displays complete health scorecards.
  - Zero Python syntax/runtime errors and verified hot-reloading behavior.
- **Todo List**:
  1. Validate full layout in Streamlit.
  2. Run automated ingestion and lineage tests.
- **Relevant Context**:
  - `streamlit_app.py:431-471`
  - `churn_app/database.py:get_data_lineage_records`
- **Status**: `[x] done`

---

## Final Delivery & Verification Sign-Off
- **Horizontal Segmented Selector**: Implemented and verified with 3 distinct full-width ingestion modes in [`streamlit_app.py`](streamlit_app.py:268).
- **Quality & Auto-Repair Flow**: Integrated with [`churn_app.faker_producer.FakerProducer`](churn_app/faker_producer.py) and [`churn_app.database.diagnose_dataframe_quality`](churn_app/database.py).
- **Lineage & Database Health**: Retained prominent full-width positioning for end-to-end auditability.
- **Production Status**: Validated and deployed.
