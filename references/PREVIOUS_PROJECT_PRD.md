# PROJECT_PRD.md

**Project Requirements Document — Master Specification**
**Version:** 1.0
**Status:** Approved for Implementation
**Prepared for:** Group 23, R R Institute of Technology, Dept. of CSE, Major Project Phase-1 (BCS685)

---

# 1. Project Overview

**Project Name (Software Implementation):** Smart Shelf — AI-Based Product Detection & Stock Monitoring System
**Original PPT Title:** "INDOOR WELLNESS – A PREDICTIVE MODEL FOR POLLUTION CONTROL USING ADVANCED AI TECHNIQUES"
**Academic Reference:** BCS685 Major Project Phase-1, Group 23 (Muskan Ranjan – 1RI23CS099, P Alekhya – 1RI23CS104, Pooja K S – 1RI23CS107), Guide: Asst. Prof. Surendra Babu M S

**Project Purpose**
This PRD defines the actual, buildable system for Group 23's major project. As explained fully in Section 2, the source PPT's title and early chapters (Chapters 1–3) describe an Indoor Air Quality (IAQ) prediction system, while its System Requirements and Expected Outcomes chapters (Chapters 4–5) describe a camera-based product/shelf monitoring system built with YOLO. This document adopts the latter as the actual implementation target, converted into a fully software-only application that runs on a normal Windows PC/laptop.

**Problem Statement (short form)**
Manual shelf monitoring in retail-like environments is slow, error-prone, and reactive — stockouts and empty shelves are often discovered only after a customer complaint or a sales loss. A computer-vision system that watches a shelf (via image, video, or webcam) and automatically reports product counts, low-stock conditions, and empty shelves can make this process faster, more consistent, and demonstrable within an academic project timeline.

**Proposed Solution**
A Python-based, YOLO-powered computer vision pipeline that ingests images, video files, or a live webcam feed of a shelf; detects and classifies products; counts them per configured shelf region; classifies each region's stock status (AVAILABLE / LOW STOCK / EMPTY) against configurable thresholds; stores detection history in a local database; and presents everything through a web-based dashboard with alerts and basic analytics. No IoT sensors, no Raspberry Pi, and no physical hardware installation are required to run or demonstrate the project.

**Target Users**
- Project evaluator / demo user (primary, for academic assessment)
- Simulated "store staff" / "inventory manager" persona (for demo storytelling and dashboard role framing)
- Administrator (configures shelves, thresholds, and product classes)

**Academic / Project Context**
This is a final-year B.E. Computer Science and Engineering major project (VTU curriculum, BCS685), evaluated in a college setting, and expected to be demonstrated live within a 5–10 minute session. The implementation must be completable by a 3-member student team without institutional hardware procurement.

**Executive Summary**
Group 23's original presentation is internally inconsistent: its narrative chapters pitch an IoT/AI air-quality prediction system, but its concrete engineering chapters (Functional Requirements, Non-Functional Requirements, Hardware/Software Requirements, and Expected Outcomes) specify a camera-and-YOLO product/shelf detection system instead. Because only the latter set of chapters contains implementable, testable engineering requirements, this PRD builds the project around them, strips out the Raspberry Pi/IoT hardware dependency mentioned incidentally in Chapter 4.3, and defines a complete, software-only "Smart Shelf" product detection and stock-monitoring application — deployable on a single Windows laptop — that an AI coding agent or developer can implement end-to-end using this document as the single source of truth.

---

# 2. PPT Requirement Analysis

## 2.1 Requirements Explicitly Stated in the PPT

**From Chapters 1–3 (Indoor Wellness narrative — NOT implemented, see Section 3):**
- Indoor Air Quality (IAQ) monitoring using IoT sensors (PM2.5, CO₂, VOC, temperature, humidity)
- Kalman filtering for sensor noise reduction
- ANN-based AQI prediction
- Threshold-based safe/unsafe classification and automated ventilation/purifier control
- Cloud integration and real-time dashboard visualization (for the IAQ concept)
- A literature review of 20 papers, all on IAQ/pollution prediction topics — none reference product detection, shelves, or YOLO

**From Chapter 4 (System Requirements — the chapter this project actually implements):**
- FR: Real-time object detection of products on shelves using a camera and a YOLO model
- FR: Stock level monitoring — available items, low stock, empty shelves
- FR: Multi-product detection — detect and classify different product types
- FR: Dashboard display of stock status
- FR: Alert generation when stock is low or shelves are empty
- NFR: Real-time performance with minimal delay
- NFR: High accuracy in identifying and counting products
- NFR: Scalability across products, users, and cameras
- NFR: Reliability with minimal downtime
- NFR: Usability for non-technical users
- NFR: Security of user data and access restriction
- NFR: Maintainability and ease of updates
- NFR: Compatibility across devices, cameras, and operating systems
- Hardware (as literally stated): Raspberry Pi 4 Model B (4 GB RAM recommended), USB camera (e.g., Logitech C270), PC/laptop with Intel i3 minimum / i5+ recommended, 4 GB RAM minimum / 8 GB recommended
- Software (as literally stated): Raspberry Pi OS or Windows, Python, YOLO (lightweight version), OpenCV, NumPy, VS Code

**From Chapter 5 (Expected Outcomes):**
- Real-time product detection on shelves via camera + YOLO with "good accuracy"
- Effective stock-level monitoring (available / low-stock / empty)
- Automatic alerts for low stock or missing items
- User-friendly dashboard showing live stock status and detection results
- Reduced manual effort and improved decision-making in inventory control

## 2.2 Requirements Implied by the PPT (not explicit, but reasonably inferred)

- Since "different types of products" must be detected and classified, some notion of product *categories/classes* is implied, even though the PPT never names specific product classes or provides a dataset.
- Since "low stock" and "empty shelf" are distinguished from "available," some counting/threshold logic per shelf or shelf-region is implied, even though the PPT gives no formula or threshold values.
- Since a "user-friendly dashboard" is required and the audience is "non-technical users," a web-based or GUI-based interface (rather than a command-line tool) is implied.
- Since "different... cameras and operating systems" compatibility is listed, a reasonably platform-independent software stack (not one hard-wired to Raspberry Pi OS only) is implied.
- Since Expected Outcomes speaks of "detection results" and "livestock status" (read as "live stock status"), some persistence/history of detections is implied even though no database is explicitly mentioned.

## 2.3 Contradictions / Inconsistencies Found — Explicitly Documented

**This is the central inconsistency in the source PPT, and it is not hidden or silently merged in this project:**

| Aspect | Chapters 1–3 (Narrative / Title) | Chapters 4–5 (Requirements / Outcomes) |
|---|---|---|
| Project title | "Indoor Wellness – A Predictive Model for Pollution Control Using Advanced AI Techniques" | Not restated, but content matches a shelf-monitoring system |
| Domain | Indoor air quality / pollution | Retail shelf / inventory monitoring |
| Sensing input | PM2.5, CO₂, VOC, temperature, humidity via IoT sensors | Camera images/video of a shelf |
| AI technique | ANN + Kalman filtering for AQI forecasting | YOLO object detection for product recognition |
| Output | Predicted AQI, ventilation ON/OFF control | Product counts, stock status, low-stock/empty-shelf alerts |
| Literature review | 20 papers, 100% on IAQ/pollution prediction | Not referenced at all in the literature review |
| Hardware | IoT sensor modules, air purifier/ventilation actuators | Raspberry Pi 4, USB webcam |
| Explicit FR/NFR chapter | None — no FRs are written for the IAQ concept | Fully written FR/NFR set describing shelf detection |

**Conclusion:** These are two structurally different systems that were not reconciled by the authors of the PPT. There is no overlap in sensors, algorithms, or outputs between the two halves of the document. This PRD does **not** attempt to merge them (e.g., it will not simulate "air quality sensors" alongside a shelf-detection system, and it will not rename shelf detection outputs as pollution metrics). Per explicit instruction, the concrete, testable Chapter 4/5 content is treated as the authoritative implementation scope, and the Indoor Wellness narrative is treated as an academic literature/motivation artifact that will be **retained only in the PRD's documentation as historical context**, not implemented.

## 2.4 Implementation Decisions Required

The following decisions had to be made because the PPT is incomplete on necessary engineering detail. Each is marked explicitly and justified:

- **Implementation Decision:** Remove Raspberry Pi as a requirement; target Windows PC/laptop only. *Why:* User's project brief explicitly forbids hardware dependency; Raspberry Pi is mentioned only once, incidentally, in Chapter 4.3, and is not needed to satisfy any FR/NFR or Expected Outcome.
- **Implementation Decision:** Use a pretrained general-purpose YOLO model (e.g., YOLOv8n/COCO-pretrained) for the initial prototype, and clearly label a custom-trained model on a project-specific dataset as the target for the final submission. *Why:* PPT says "YOLO (lightweight version)" but specifies no dataset or product classes; a generic pretrained model cannot recognize arbitrary retail products (see Section 11).
- **Implementation Decision:** Define concrete stock-level thresholds (e.g., count-based bands) as configurable, not hard-coded. *Why:* PPT states the system must distinguish "available / low stock / empty" but gives no numeric definition.
- **Implementation Decision:** Define shelf regions using manually configured Regions of Interest (ROIs). *Why:* PPT does not explain how the system knows where a "shelf" is in the camera frame; pure CV cannot infer shelf boundaries without configuration or a trained shelf-segmentation model.
- **Implementation Decision:** Use SQLite for data storage. *Why:* PPT does not mention a database, but "dashboard," "alerts," and "history/analytics" outcomes are not achievable without persistence; SQLite requires no separate server and matches "simple, robust technology" instruction.
- **Implementation Decision:** Use FastAPI (Python) as the backend and a lightweight HTML/CSS/JS (or a small React) frontend as the dashboard. *Why:* PPT doesn't specify a framework; Python was explicitly listed, and FastAPI/Flask were explicitly suggested by the user's brief.
- **Implementation Decision:** Alerts are in-dashboard/toast only for v1; no email/SMS integration. *Why:* PPT does not mention alert channels beyond "generate alerts"; user's brief instructs avoiding unnecessary external services in the first version.
- **Implementation Decision:** No claim of a specific accuracy number until the model is actually trained and evaluated on a held-out test set. *Why:* Explicit rule from user; PPT's "good accuracy" is not a number.

---

# 3. Final Implementation Scope

## 3.1 In Scope

- Software-only computer-vision pipeline for product detection using YOLO
- Product classification into a configurable set of product classes
- Product counting per detection pass and per shelf region
- Shelf/stock-status classification: AVAILABLE / LOW STOCK / EMPTY
- Low-stock and empty-shelf detection logic based on configurable thresholds and ROIs
- Multi-product (multiple classes, multiple instances) detection in a single frame
- Input support: static image upload, video file upload, and live webcam feed
- Web-based dashboard (desktop-browser-first, works locally on Windows)
- Alert system (in-dashboard + toast notifications)
- Local relational database (SQLite) for detections, inventory state, shelves, and alerts
- Basic analytics (trends, most/least detected products, alert history)
- Configuration UI/config file for thresholds, shelf ROIs, and product classes
- Model management (load pretrained baseline model; support swapping in a custom-trained model file)
- Error handling for missing model, bad input, camera failure, DB failure, low-confidence detections
- Local deployment runnable entirely on a single Windows laptop with no internet dependency after setup

## 3.2 Out of Scope

- Real IoT sensor integration of any kind (PM2.5, CO₂, VOC, temperature, humidity sensors)
- Kalman filtering of sensor telemetry (not applicable — there is no sensor telemetry in this implementation)
- ANN-based air-quality prediction / AQI forecasting
- Physical ventilation, HVAC, or air-purifier control/actuation
- Raspberry Pi or any embedded/edge hardware requirement
- Multi-camera synchronized deployments (single active camera/input source per session for v1; listed only as a future enhancement)
- Cloud deployment, cloud storage, or cloud AI services (local-only for v1)
- Integration with real POS/ERP/retail systems
- Mobile application

## 3.3 Hardware Excluded

No Raspberry Pi, no external sensor boards/modules, no relay/actuator hardware for ventilation, no dedicated GPU requirement (CPU inference is acceptable for prototype scale, though a GPU will improve frame rate if available). A single built-in or USB webcam is optional and only needed for the live-camera input mode — the system is fully demonstrable using pre-recorded images/video without any camera at all.

## 3.4 IoT Excluded

No sensor telemetry ingestion, no MQTT/CoAP/IoT message brokers, no device-provisioning or sensor-calibration workflows of any kind.

## 3.5 Physical Sensors Excluded

No PM2.5, CO₂, VOC, temperature, or humidity sensors, physical or simulated. The system will not generate synthetic/fake sensor readings to preserve the original "Indoor Wellness" narrative — per explicit instruction, this would misrepresent the deliverable.

---

# 4. Problem Statement

Retail and warehouse environments rely heavily on manual, periodic visual inspection to determine whether shelves are adequately stocked. This process does not scale well with the number of shelves, is inconsistent between staff members, and typically detects stockouts only after they have already impacted sales or customer experience. Existing basic camera-based systems in this space (as summarized in the "Existing System" framing style used by the PPT for its own domain) tend to be non-interactive, provide raw video only, and lack automated classification, counting, or alerting.

There is a need for a lightweight, camera-driven software system that can be pointed at a shelf — via a live webcam, an uploaded video, or a set of images — and automatically (a) detect the presence and type of products, (b) count them, (c) determine whether the shelf is adequately stocked, running low, or empty, and (d) surface this information through a real-time dashboard with alerting, without requiring specialized hardware, sensor installation, or cloud infrastructure. This project addresses that need by building a YOLO-based computer-vision pipeline packaged as a self-contained Windows-deployable application.

---

# 5. Objectives

Derived from the PPT's Chapter 4/5 functional requirements and expected outcomes, converted into measurable objectives:

1. **O1 — Detection:** Achieve product detection on static test images with a measured precision/recall reported at project completion (target guideline: mAP@0.5 ≥ 0.6 on the custom validation set; exact number to be reported, not assumed).
2. **O2 — Classification:** Correctly assign each detected product to one of the configured product classes on the validation set, with per-class accuracy reported.
3. **O3 — Counting:** Produce a per-frame and per-shelf-region product count that matches manual ground-truth counts within a documented tolerance during test scenarios.
4. **O4 — Stock Classification:** Automatically classify each configured shelf region into AVAILABLE / LOW STOCK / EMPTY using the documented threshold logic (Section 13), verified against at least 3 manually staged scenarios (full, partially empty, empty).
5. **O5 — Real-Time Operation:** Process live webcam frames at a usable interactive rate on the target laptop spec (guideline: ≥ 5 FPS on CPU with a "nano/small" YOLO variant; report actual measured FPS).
6. **O6 — Dashboard & Alerts:** Deliver a working dashboard that reflects current stock status and issues a visible alert within a few seconds of a low-stock/empty-shelf condition being detected.
7. **O7 — History & Analytics:** Persist every detection run and alert event, and present at least one trend chart (e.g., stock status over time) in the dashboard.
8. **O8 — Demonstrability:** Complete a full end-to-end demo (Section 27) in under 10 minutes without manual code intervention.

---

# 6. Target Users

- **Project Evaluator / Demo User** — the primary audience during the academic evaluation; needs a clear, guided, visually convincing walkthrough.
- **Inventory Manager (simulated role)** — the persona the dashboard is designed for; represents someone who needs an at-a-glance view of stock health and alerts.
- **Administrator (simulated role)** — configures product classes, shelf ROIs, and stock thresholds; may be the same person as the demo user during evaluation.

*(Store staff is intentionally not modeled as a separate role in v1 — the Inventory Manager and Administrator roles cover the dashboard's real needs without unnecessarily expanding scope.)*

---

# 7. Functional Requirements

Each requirement includes Description, Input, Processing, Output, and Acceptance Criteria.

### FR-01 Product Detection
- **Description:** The system shall detect products present in a given image frame using a YOLO object detection model.
- **Input:** A single image frame (from image upload, video frame, or webcam frame).
- **Processing:** Frame is preprocessed (resize/normalize) and passed through the YOLO model; raw detections are filtered by a confidence threshold and refined with Non-Maximum Suppression (NMS).
- **Output:** A list of bounding boxes, each with class label and confidence score.
- **Acceptance Criteria:** Given a test image containing known products, the system returns at least one bounding box per clearly visible, unoccluded product above the configured confidence threshold.

### FR-02 Product Classification
- **Description:** Each detected object shall be classified into one of the configured product classes.
- **Input:** Cropped region / detection output from FR-01.
- **Processing:** The YOLO model's class head assigns a class label and confidence to each detection.
- **Output:** Class label + confidence per detection.
- **Acceptance Criteria:** For a labeled test set, class-level accuracy is measured and reported (no specific number assumed in advance — see Section 23).

### FR-03 Product Counting
- **Description:** The system shall count the number of detected products, both in total and per class, for a given frame or shelf region.
- **Input:** Detection list from FR-01/FR-02.
- **Processing:** Aggregate detections by class and by configured shelf ROI.
- **Output:** Total count, per-class count, per-shelf-region count.
- **Acceptance Criteria:** For a staged test image with a known number of items, system count matches manual count within an agreed tolerance (documented at test time).

### FR-04 Shelf Monitoring
- **Description:** The system shall associate detections with one or more configured shelf regions (ROIs) and track occupancy over time.
- **Input:** Detection list + configured ROI polygon/box coordinates.
- **Processing:** Determine which detections fall inside which ROI; compute per-ROI occupancy metrics.
- **Output:** Per-shelf detection count and occupancy percentage.
- **Acceptance Criteria:** Given 2+ configured shelf ROIs in one frame, the system reports counts independently for each region.

### FR-05 Stock-Level Classification
- **Description:** The system shall classify each shelf region's stock level as AVAILABLE, LOW STOCK, or EMPTY.
- **Input:** Per-shelf detection count/occupancy from FR-04, configured thresholds.
- **Processing:** Apply the threshold logic defined in Section 13.
- **Output:** A status label per shelf region.
- **Acceptance Criteria:** Staged test scenarios (full shelf, half-stocked shelf, empty shelf) each produce the expected status label.

### FR-06 Low-Stock Detection
- **Description:** The system shall flag a shelf region as LOW STOCK when detected count/occupancy falls between the empty and available thresholds.
- **Input:** Stock status from FR-05.
- **Processing:** Threshold comparison (Section 13).
- **Output:** Boolean low-stock flag + numeric value that triggered it.
- **Acceptance Criteria:** A staged partially-stocked shelf is correctly flagged as LOW STOCK, not AVAILABLE or EMPTY.

### FR-07 Empty-Shelf Detection
- **Description:** The system shall flag a shelf region as EMPTY when no (or near-zero) products are detected within it for a sustained check.
- **Input:** Per-shelf detection count from FR-04.
- **Processing:** Compare count/occupancy to the empty threshold (Section 14); optionally require the condition to persist across N consecutive frames for video/live input to reduce false positives from momentary occlusion.
- **Output:** Boolean empty-shelf flag.
- **Acceptance Criteria:** A staged empty shelf region is flagged EMPTY within the configured confirmation window, without flickering on single-frame noise.

### FR-08 Multi-Product Detection
- **Description:** The system shall detect and correctly separate multiple product types/instances present simultaneously in one frame.
- **Input:** Frame containing 2+ product classes.
- **Processing:** YOLO's multi-class, multi-instance detection output.
- **Output:** Distinct bounding boxes/labels for each instance and class.
- **Acceptance Criteria:** A test frame with at least 2 different classes and 2+ instances of at least one class is correctly detected as multiple distinct boxes with correct classes.

### FR-09 Image Input
- **Description:** The system shall accept a single uploaded image (JPG/PNG) as a detection input.
- **Input:** Image file via dashboard upload control.
- **Processing:** Validate file type/size; run detection pipeline once.
- **Output:** Annotated image + detection/stock results.
- **Acceptance Criteria:** Uploading a valid image returns results within a few seconds; invalid file types are rejected with a clear message.

### FR-10 Video Input
- **Description:** The system shall accept an uploaded video file and run detection across its frames (sampled at a configurable interval).
- **Input:** Video file (e.g., MP4).
- **Processing:** Extract frames at a configured sampling rate; run detection per sampled frame; aggregate results across the video.
- **Output:** Per-frame results plus a summary (e.g., final/most common stock status).
- **Acceptance Criteria:** Uploading a short test video produces frame-by-frame and summary results without crashing.

### FR-11 Live Camera Input
- **Description:** The system shall support a live webcam feed as a continuous detection input.
- **Input:** Webcam device stream.
- **Processing:** Continuously capture frames, run detection at a configurable interval, and update dashboard state in near real time.
- **Output:** Live-updating detection overlay + live stock status.
- **Acceptance Criteria:** With a webcam connected, starting "Live Mode" updates the dashboard's detection view within the configured refresh interval; if no webcam is present, the system reports a clear camera-unavailable message instead of crashing (see FR-18/Section 25).

### FR-12 Dashboard
- **Description:** The system shall present a web-based dashboard summarizing current and historical detection/stock state.
- **Input:** Aggregated data from the database and live detection pipeline.
- **Processing:** Query and format data for display (see Section 15 for exact components).
- **Output:** Rendered dashboard page.
- **Acceptance Criteria:** All components listed in Section 15 render with real data from at least one completed detection run.

### FR-13 Alerts
- **Description:** The system shall generate an alert when a shelf region transitions into LOW STOCK or EMPTY status, or when a system-level failure occurs.
- **Input:** Stock status changes (FR-05/06/07), system health events.
- **Processing:** Compare new status to previous status; if a new alert-worthy condition is reached, create an alert record and surface it in the UI.
- **Output:** Alert entries visible in-dashboard and as toast notifications.
- **Acceptance Criteria:** Triggering a staged low-stock/empty condition produces a visible alert within a few seconds, and the alert is retrievable later from alert history.

### FR-14 Inventory History
- **Description:** The system shall persist every detection run's results (counts, statuses, timestamps) for later review.
- **Input:** Completed detection run output.
- **Processing:** Write structured records to the database (Section 17).
- **Output:** Queryable history accessible via the Inventory/Analytics pages.
- **Acceptance Criteria:** After several detection runs, the history page lists them in chronological order with correct data.

### FR-15 Analytics
- **Description:** The system shall provide basic trend analytics derived from inventory history.
- **Input:** Historical detection/alert records.
- **Processing:** Aggregate by time, product class, or shelf (e.g., counts over time, alert frequency).
- **Output:** At least one chart (e.g., line/bar chart) and summary statistics.
- **Acceptance Criteria:** After 5+ historical detection runs exist, the analytics page renders a non-empty, correctly computed chart.

### FR-16 Configuration
- **Description:** The system shall allow an administrator to configure product classes, shelf ROIs, and stock thresholds without code changes.
- **Input:** Settings form or configuration file.
- **Processing:** Validate and persist configuration; apply on next detection run.
- **Output:** Updated configuration in effect.
- **Acceptance Criteria:** Changing a threshold value in Settings changes the stock-status outcome of a subsequent identical detection run.

### FR-17 Model Management
- **Description:** The system shall allow the active YOLO model file to be swapped (e.g., from the pretrained baseline to a custom-trained model) via configuration.
- **Input:** Model file path/selection in Settings.
- **Processing:** Load the specified model weights at startup or on demand; validate the file loads successfully.
- **Output:** Active model info displayed in the dashboard (e.g., "Model Info" panel).
- **Acceptance Criteria:** Switching the configured model path and restarting the detection service causes the new model to be used, and the dashboard reflects the active model's name/version.

### FR-18 Error Handling
- **Description:** The system shall handle predictable failure conditions gracefully rather than crashing (see full list in Section 25).
- **Input:** Any of the failure conditions in Section 25.
- **Processing:** Catch the specific error, log it, and return/display a clear, user-readable message.
- **Output:** Graceful degradation + visible error/alert state.
- **Acceptance Criteria:** Each failure condition in Section 25 is manually tested and confirmed to not crash the application.

---

# 8. Non-Functional Requirements

| Category | Requirement | Realistic Target |
|---|---|---|
| **Performance** | Detection pipeline shall run with minimal delay for interactive use. | Image mode: result within ~2–5 sec on CPU. Live mode: aim for ≥ 5 FPS with a nano/small YOLO variant on a mid-range laptop CPU; report actual measured FPS at project completion. |
| **Accuracy** | Detection/classification model should achieve high accuracy on the project's own validation set. | No accuracy is assumed or advertised until measured (Section 23); do not claim 100% under any circumstance. |
| **Scalability** | The system should handle a growing number of product classes, shelf regions, and stored history records without redesign. | Architecture supports adding classes/ROIs via configuration, not code changes. SQLite is adequate up to tens of thousands of history rows for a student project; migration path to PostgreSQL noted for future scale (Section 30). |
| **Reliability** | The system should remain operable through recoverable errors (bad frame, brief camera glitch) without a full crash. | No unhandled exceptions should terminate the running application during a standard demo session. |
| **Usability** | Non-technical users (e.g., an evaluator) should be able to operate the dashboard without a manual. | Primary actions (upload image/video, start live mode, view alerts) reachable within 2 clicks from the dashboard home. |
| **Security** | Protect against trivial misuse; restrict configuration/administration to an authenticated context. | Basic login for the Settings/Admin area (Section 24); no requirement for enterprise-grade security given academic scope. |
| **Maintainability** | Code should be modular so a component (e.g., swapping YOLO version) can be updated independently. | Clear separation of Input, Processing/AI, Application, Data, and Presentation layers (Section 9); documented configuration rather than hard-coded constants. |
| **Compatibility** | The system should work on standard Windows laptops with common USB webcams, without OS-specific hacks. | Verified on Windows 10/11 with Python 3.10+; camera access via OpenCV's standard VideoCapture API. |
| **Availability** | The local application should be available throughout a demo session without needing restarts. | Backend process runs continuously for the duration of a typical demo (10–30 min) without memory leaks causing failure. |
| **Explainability** | Detection results should be interpretable, not a black box, for academic evaluation purposes. | Bounding boxes with class + confidence are always shown; stock-status decisions are traceable to a specific count/threshold shown in the UI. |

---

# 9. System Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                             │
│  Image Upload   |   Video Upload   |   Live Webcam Stream      │
└───────────────────────────┬───────────────────────────────────┘
                             │ raw frame(s)
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                     PROCESSING LAYER                            │
│  OpenCV frame capture/decoding → preprocessing                  │
│  (resize, color conversion, normalization) → YOLO inference     │
└───────────────────────────┬───────────────────────────────────┘
                             │ raw detections (boxes, classes, scores)
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                          AI LAYER                                │
│  NMS filtering → per-class product counting →                   │
│  shelf ROI association → occupancy computation                  │
└───────────────────────────┬───────────────────────────────────┘
                             │ structured detection result
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                            │
│  Stock threshold logic (AVAILABLE/LOW/EMPTY) → alert logic →     │
│  history recording → analytics aggregation → configuration mgmt │
└───────────────────────────┬───────────────────────────────────┘
                             │ persisted + computed data
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                │
│  SQLite database: products, shelves, detections, inventory,       │
│  alerts, system_settings                                          │
└───────────────────────────┬───────────────────────────────────┘
                             │ REST API (FastAPI)
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                              │
│  Web dashboard: Dashboard, Detection, Inventory, Alerts,           │
│  Analytics, Settings, Model Info pages                             │
└───────────────────────────────────────────────────────────────┘
```

**Component explanations:**
- **Input Layer:** Accepts three input modes; abstracted behind a common "FrameSource" interface so the rest of the pipeline doesn't care whether a frame came from an image, a video, or a webcam.
- **Processing Layer:** Pure computer-vision preprocessing and model inference; no business logic here — its only job is "frame in, raw detections out."
- **AI Layer:** Turns raw detections into meaningful, shelf-aware structured data (counts per class, counts per ROI, occupancy percentage). This is where NMS and ROI-association logic live.
- **Application Layer:** Owns all business rules — what counts as "low stock," when to fire an alert, how history is recorded, how config changes are applied. This layer is what actually encodes the FR/NFRs.
- **Data Layer:** A single local SQLite file; no external DB server needed, matching the "runs on a normal Windows laptop" constraint.
- **Presentation Layer:** A FastAPI backend serving a REST API, plus a browser-based frontend that polls/queries this API to render the dashboard.

---

# 10. Recommended Technology Stack

| Layer | Technology | Justification |
|---|---|---|
| Language | Python 3.10+ | Explicitly specified in PPT; best-supported language for YOLO/OpenCV ecosystem. |
| Object Detection | Ultralytics YOLOv8 (nano/small variant) | Actively maintained, easy pip install, good Windows/CPU support, supports both pretrained (COCO) and custom training. Matches PPT's "YOLO (lightweight version)." |
| Computer Vision | OpenCV (`opencv-python`) | Explicitly specified in PPT; handles frame capture, preprocessing, drawing overlays. |
| Numeric processing | NumPy | Explicitly specified in PPT. |
| Backend/API | FastAPI + Uvicorn | Explicitly suggested in brief; async-friendly, auto-generates API docs, integrates cleanly with a Python CV pipeline (vs. needing a separate service for a different language). |
| Database | SQLite (via SQLAlchemy or raw `sqlite3`) | Zero-install, file-based, explicitly suggested; adequate for the project's data scale. |
| Frontend | HTML/CSS/JavaScript (server-rendered templates or a lightweight SPA) + Chart.js | No build-tooling overhead, quick to demo, avoids unnecessary complexity of a full SPA framework for a student project; Chart.js is simple and widely documented for analytics charts. |
| Dev environment | VS Code | Explicitly specified in PPT. |
| OS Target | Windows 10/11 | Matches "normal Windows laptop" requirement; Raspberry Pi OS target is dropped per Section 3.3. |

*Note:* A small React frontend is an acceptable alternative to server-rendered HTML if the team is comfortable with it, but plain HTML/CSS/JS is recommended as the default for simplicity, installation ease, and AI-agent implementability, per the user's brief.

---

# 11. AI / Computer Vision Design

- **YOLO's role:** YOLO (You Only Look Once) is a single-pass convolutional object detector that predicts bounding boxes and class probabilities directly from an image in one forward pass, making it suitable for near-real-time use on modest hardware.
- **Object detection:** For each input frame, the model outputs a set of candidate bounding boxes, each with a class label and confidence score.
- **Product classification:** Classification is not a separate step — YOLO's detection head jointly performs localization and classification in one pass; the class assigned to each box is the product classification result.
- **Bounding boxes:** Represented as (x, y, width, height) or (x1, y1, x2, y2) coordinates around each detected product.
- **Confidence scores:** A 0–1 score reflecting the model's certainty; detections below a configurable confidence threshold (e.g., 0.4–0.5, tunable) are discarded.
- **Non-Maximum Suppression (NMS):** Removes duplicate/overlapping boxes for the same object by keeping the highest-confidence box among heavily overlapping candidates (IoU threshold configurable, e.g., 0.45).
- **Product counting:** Simple aggregation — count of surviving detections, grouped by class and by shelf ROI.
- **Shelf region analysis:** Each configured ROI (a rectangle/polygon over the frame) is checked for which detections' centers fall inside it; per-ROI counts feed the stock-level logic.
- **Low-stock / empty-shelf logic:** See Sections 13–14.

**IMPORTANT — Pretrained vs. Custom Model Distinction (explicitly required by the brief):**

**A. Prototype phase — pretrained/general model.**
A YOLOv8 model pretrained on COCO (80 general object classes: e.g., bottle, cup, book, etc.) will be used for early development and pipeline testing. **This pretrained model cannot reliably recognize arbitrary supermarket/retail products** (e.g., specific packaged-goods brands, specific SKU packaging) because those classes do not exist in COCO's training data. It is used only to validate that the detection → counting → dashboard pipeline works end-to-end, using whatever COCO classes are visually present (e.g., bottles, boxes) as a stand-in for "products."

**B. Final project — custom-trained model.**
For the final submission, the team must collect and annotate a small custom dataset representing the actual "products" they intend to demo (e.g., a defined set of 3–8 distinct packaged items, such as different snack boxes or bottles) and fine-tune/train a YOLOv8 model on these classes (see Section 12). Only after this training step is complete may the project claim genuine "product detection and classification" for those specific classes. No claim of trained-model accuracy is made until this training and evaluation actually happens.

---

# 12. Dataset Strategy

- **Dataset collection:** Capture 150–300+ images per product class using a phone or laptop webcam, varying angle, lighting, distance, and shelf arrangement (single item, multiple items, partially occluded, empty shelf).
- **Product classes:** Team selects 3–8 distinct, visually distinguishable physical products available to them (e.g., specific snack packets, bottles, boxes) — kept small and achievable for a student timeline. *(Implementation Decision — PPT does not name classes.)*
- **Image requirements:** Minimum 224×224 effective resolution on the product region; mixed backgrounds recommended to avoid the model learning the background instead of the product; include some empty-shelf images as a "no product" scenario.
- **Annotation:** Bounding-box annotation in YOLO format (class, x_center, y_center, width, height, normalized). Recommended free tools: **LabelImg** or **Roboflow** (Roboflow also handles augmentation and format export automatically and has a generous free tier suitable for student projects).
- **Train/validation/test split:** Recommended 70% / 20% / 10%, stratified per class where possible.
- **Data augmentation:** Standard YOLO training augmentations (mosaic, horizontal flip, brightness/contrast jitter, slight rotation) — Ultralytics YOLOv8 applies sensible defaults automatically; avoid unnecessary custom augmentation code.
- **Dataset versioning:** Keep dataset in a clearly versioned folder (e.g., `datasets/v1/`, `datasets/v2/`) or use Roboflow's built-in versioning; record which dataset version trained which model file.
- **Model training:** Fine-tune a pretrained YOLOv8n/s checkpoint on the custom dataset using Ultralytics' training CLI/API; train on available hardware (CPU is workable for a small dataset/short training run, but a free-tier GPU notebook, e.g., Google Colab, is recommended to keep training time reasonable).
- **Model evaluation:** Evaluate on the held-out test split using the metrics in Section 23; only report numbers actually produced by this evaluation.

---

# 13. Stock-Level Logic

Thresholds are **configurable**, not hard-coded, and are set per shelf region (since different shelves may reasonably hold different maximum capacities).

**Definitions:**
- `expected_capacity` — the configured "full shelf" reference count for a given shelf ROI (set once during shelf configuration, e.g., by counting products when the shelf is fully stocked).
- `detected_count` — the number of product detections whose center point falls inside the shelf ROI in the current frame/analysis window.
- `occupancy_ratio = detected_count / expected_capacity` (capped at 1.0).

**Classification algorithm (default, fully configurable):**

```
IF detected_count == 0 (or occupancy_ratio <= empty_threshold):
    STATUS = EMPTY
ELSE IF occupancy_ratio <= low_stock_threshold:
    STATUS = LOW STOCK
ELSE:
    STATUS = AVAILABLE
```

**Default suggested values (tunable in Settings, not claimed as universally correct):**
- `empty_threshold` = 0.05 (i.e., ≤ 5% of expected capacity, effectively "nothing there")
- `low_stock_threshold` = 0.35 (i.e., ≤ 35% of expected capacity)

**Why configurable:** Different shelves, camera angles, and product sizes will need different practical thresholds; hard-coding a single number for all shelves would misrepresent real-world variability and was explicitly disallowed unless justified. Thresholds are stored in the `system_settings` table (Section 17) and editable via the Settings page (FR-16).

---

# 14. Empty Shelf Detection

**Limitations of pure computer-vision empty-shelf detection (stated honestly, per instruction):**
- The system cannot know, out of the box, where a "shelf" is in the camera frame — it has no innate concept of shelf geometry. It only knows what is inside a manually configured ROI.
- The system cannot know what "full" looks like for a shelf it has never seen configured — `expected_capacity` must be set once, manually, by an administrator during setup.
- Occlusion (e.g., a person's hand or cart briefly in front of the shelf) can cause a false "empty" reading in a single frame.
- Poor lighting or a camera angle that makes background visible behind sparse items can confuse a generic pretrained model (this is one more reason the custom-trained model in Section 11-B is important for real accuracy).
- A shelf that has non-product items placed on it (during a demo, anything not in the trained class list) will not be detected at all, which could be misread as "empty" even though something is physically there — this is documented as a known limitation, not silently hidden.

**Practical implementation approach:**
1. **Shelf ROI configuration:** Administrator draws or numerically enters a bounding rectangle/polygon over the camera frame per shelf, via the Settings/Configuration UI, once during setup.
2. **Expected product presence:** Administrator sets `expected_capacity` for that ROI (e.g., by counting items on a fully stocked shelf once, or entering a known number).
3. **Detection count:** For each analyzed frame, count detections whose center falls within the ROI.
4. **Region occupancy:** Compute `occupancy_ratio` as defined in Section 13.
5. **Configurable thresholds:** Apply the Section 13 logic to classify AVAILABLE / LOW STOCK / EMPTY.
6. **Temporal confirmation (for live/video mode):** To reduce false "empty" flickers from momentary occlusion, require the EMPTY condition to hold for N consecutive analyzed frames (configurable, default N = 3) before firing an alert.

---

# 15. Dashboard Requirements

| Component | Definition |
|---|---|
| **Total Products Detected** | Running count of all detections in the most recent analysis run/session. |
| **Product Categories** | List/breakdown of configured product classes and per-class counts from the latest run. |
| **Current Stock Status** | Per-shelf-region status badge: AVAILABLE (green) / LOW STOCK (amber) / EMPTY (red). |
| **Low-Stock Products** | A filtered list of shelf regions/classes currently flagged LOW STOCK. |
| **Empty Shelves** | A filtered list of shelf regions currently flagged EMPTY. |
| **Detection Confidence** | Average/min confidence score of detections in the latest run, shown per shelf or overall. |
| **Camera/Input Status** | Indicator of whether the active input source (webcam/video/image) is currently connected/active or has an error. |
| **Recent Detections** | A scrollable list/table of the last N detection events with timestamp, shelf, counts, and status. |
| **Inventory History** | A dedicated page/table of all past detection runs, filterable by date/shelf/class. |
| **Alerts** | A panel listing active and historical alerts, with severity, timestamp, and the condition that triggered them. |
| **Charts** | At least one time-series chart (e.g., stock status/occupancy over time) and one categorical chart (e.g., count per product class) using Chart.js. |
| **System Status** | Health indicators: active model name/version, database connectivity, last successful detection run timestamp. |

---

# 16. Alert System

| Alert Type | Trigger | Channel (v1) |
|---|---|---|
| Low-Stock Alert | A shelf region's status transitions to LOW STOCK | In-dashboard alert panel + toast notification |
| Empty-Shelf Alert | A shelf region's status transitions to EMPTY (after temporal confirmation, Section 14) | In-dashboard alert panel + toast notification |
| Detection Failure Alert | The detection pipeline throws an unrecoverable error for a given input | In-dashboard alert panel |
| Camera/Input Failure | Webcam device not found, disconnected, or video file corrupted | In-dashboard alert panel + status indicator (Section 15) |
| Model Failure | Configured model file fails to load or is missing | In-dashboard alert panel + blocks new detection runs until resolved |

**Decision:** All alerts are **in-dashboard + toast notification only** for v1. Browser push notifications and email are explicitly excluded from v1 per the brief's instruction to avoid unnecessary external services; both are listed as Future Enhancements (Section 30) if the team wants to extend the project later.

---

# 17. Database Design

**Engine:** SQLite (file-based, e.g., `smartshelf.db`)

### `products`
| Field | Type | Purpose | Key |
|---|---|---|---|
| product_id | INTEGER | Unique identifier | PK |
| name | TEXT | Product class name (matches YOLO class label) | |
| category | TEXT | Optional grouping label | |
| created_at | DATETIME | Record creation timestamp | |

### `shelves`
| Field | Type | Purpose | Key |
|---|---|---|---|
| shelf_id | INTEGER | Unique identifier | PK |
| name | TEXT | Human-readable shelf label (e.g., "Aisle 1 - Shelf A") | |
| roi_coordinates | TEXT (JSON) | Stored ROI polygon/box coordinates | |
| expected_capacity | INTEGER | Reference "full shelf" count | |
| low_stock_threshold | REAL | Fractional threshold (Section 13) | |
| empty_threshold | REAL | Fractional threshold (Section 13) | |
| created_at | DATETIME | Record creation timestamp | |

### `detections`
| Field | Type | Purpose | Key |
|---|---|---|---|
| detection_id | INTEGER | Unique identifier | PK |
| run_id | INTEGER | Groups detections from the same analysis run | FK → `detection_runs.run_id` |
| shelf_id | INTEGER | Which shelf region this detection belongs to (nullable if outside any ROI) | FK → `shelves.shelf_id` |
| product_id | INTEGER | Detected product class | FK → `products.product_id` |
| confidence | REAL | Detection confidence score | |
| bbox_x, bbox_y, bbox_w, bbox_h | REAL | Bounding box coordinates | |
| detected_at | DATETIME | Timestamp of this detection | |

### `detection_runs`
| Field | Type | Purpose | Key |
|---|---|---|---|
| run_id | INTEGER | Unique identifier | PK |
| input_type | TEXT | 'image' / 'video' / 'live' | |
| source_reference | TEXT | Filename or session identifier | |
| model_version | TEXT | Active model file/name used | |
| started_at | DATETIME | Run start time | |
| completed_at | DATETIME | Run completion time | |

### `inventory`
| Field | Type | Purpose | Key |
|---|---|---|---|
| inventory_id | INTEGER | Unique identifier | PK |
| shelf_id | INTEGER | Which shelf this status belongs to | FK → `shelves.shelf_id` |
| run_id | INTEGER | Which run produced this status | FK → `detection_runs.run_id` |
| detected_count | INTEGER | Count of detections in this shelf for this run | |
| occupancy_ratio | REAL | Computed occupancy | |
| status | TEXT | AVAILABLE / LOW STOCK / EMPTY | |
| recorded_at | DATETIME | Timestamp | |

### `alerts`
| Field | Type | Purpose | Key |
|---|---|---|---|
| alert_id | INTEGER | Unique identifier | PK |
| alert_type | TEXT | LOW_STOCK / EMPTY_SHELF / DETECTION_FAILURE / CAMERA_FAILURE / MODEL_FAILURE | |
| shelf_id | INTEGER | Related shelf, if applicable (nullable) | FK → `shelves.shelf_id` |
| message | TEXT | Human-readable alert message | |
| severity | TEXT | INFO / WARNING / CRITICAL | |
| is_resolved | BOOLEAN | Whether the alert condition has cleared | |
| created_at | DATETIME | Timestamp raised | |
| resolved_at | DATETIME | Timestamp cleared (nullable) | |

### `system_settings`
| Field | Type | Purpose | Key |
|---|---|---|---|
| setting_key | TEXT | Setting name (e.g., `confidence_threshold`, `nms_iou_threshold`, `active_model_path`) | PK |
| setting_value | TEXT | Setting value (stored as text, parsed by type at runtime) | |
| updated_at | DATETIME | Last modified timestamp | |

---

# 18. API Design

Backend: FastAPI. Base path: `/api`

| Method | Path | Purpose | Request | Response | Errors |
|---|---|---|---|---|---|
| POST | `/api/detect/image` | Run detection on an uploaded image | multipart file | Annotated image URL + detection/stock JSON | 400 invalid file, 500 model error |
| POST | `/api/detect/video` | Run detection on an uploaded video | multipart file | Per-frame + summary JSON | 400 invalid file, 500 processing error |
| POST | `/api/live/start` | Start live webcam detection session | `{camera_index}` | `{session_id, status}` | 503 camera unavailable |
| POST | `/api/live/stop` | Stop the active live session | `{session_id}` | `{status}` | 404 session not found |
| GET | `/api/live/{session_id}/status` | Poll current live detection state | — | Latest detection/stock JSON | 404 session not found |
| GET | `/api/shelves` | List all configured shelf regions | — | Array of shelf objects | — |
| POST | `/api/shelves` | Create a new shelf region config | Shelf ROI + thresholds | Created shelf object | 400 invalid ROI |
| PUT | `/api/shelves/{shelf_id}` | Update a shelf's config/thresholds | Partial shelf object | Updated shelf object | 404 not found |
| DELETE | `/api/shelves/{shelf_id}` | Remove a shelf region | — | `{status}` | 404 not found |
| GET | `/api/products` | List configured product classes | — | Array of product objects | — |
| POST | `/api/products` | Add a new product class mapping | `{name, category}` | Created product object | 400 duplicate name |
| GET | `/api/inventory` | Get current inventory status for all shelves | Query: `shelf_id` (optional) | Array of latest inventory statuses | — |
| GET | `/api/history/runs` | List past detection runs | Query: date range, input_type | Array of run summaries | — |
| GET | `/api/history/runs/{run_id}` | Get details of a specific run | — | Full run + detections | 404 not found |
| GET | `/api/alerts` | List alerts | Query: `resolved`, `severity` | Array of alert objects | — |
| POST | `/api/alerts/{alert_id}/resolve` | Mark an alert resolved | — | Updated alert object | 404 not found |
| GET | `/api/analytics/summary` | Aggregated analytics for dashboard charts | Query: date range | Chart-ready JSON | — |
| GET | `/api/settings` | Get current system settings | — | Key/value settings object | — |
| PUT | `/api/settings` | Update system settings (thresholds, model path, etc.) | Partial settings object | Updated settings object | 400 invalid value |
| GET | `/api/model/info` | Get active model metadata | — | `{model_path, version, classes}` | 500 model not loaded |

---

# 19. UI/UX Requirements

**Navigation:** Persistent left sidebar with icons + labels: Dashboard, Detection, Inventory, Alerts, Analytics, Settings, Model Info. Top bar shows current system status (camera/model/DB health) and a notification bell for active alerts.

- **Dashboard (home):** Summary cards (Section 15 components) + at-a-glance shelf status grid + recent detections table + primary charts.
- **Detection page:** Tabs for Image Upload / Video Upload / Live Camera; shows the annotated frame with bounding boxes overlaid live; shows per-class counts and per-shelf status immediately below the frame.
- **Inventory page:** Table/grid of all configured shelves with current status, capacity, and a "view history" drill-down per shelf.
- **Alerts page:** Filterable list of alerts (active/resolved, by severity/type), with a "mark resolved" action.
- **Analytics page:** Time-series chart of stock status/occupancy, bar chart of detections per product class, alert-frequency chart.
- **Settings page:** Forms for shelf ROI configuration (with a frame preview to draw/adjust the ROI box), threshold sliders, product class management, active model path selection, and basic authentication credentials.
- **Model Information page:** Displays active model file name, whether it is the pretrained baseline or a custom-trained model, class list, and last-evaluated metrics (Section 23), if available.

**Visual tone:** Clean, modern dashboard aesthetic (card-based layout, clear status colors — green/amber/red — legible typography, adequate whitespace) so it presents as a credible software product during evaluation rather than a bare debug UI.

---

# 20. Project Folder Structure

```
smart-shelf/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app entrypoint
│   │   ├── api/                    # API route modules (detect, shelves, alerts, etc.)
│   │   ├── core/                   # Config loading, settings management
│   │   ├── cv/                     # OpenCV + YOLO inference pipeline
│   │   ├── logic/                  # Stock threshold logic, alert logic
│   │   ├── db/                     # SQLAlchemy models, session, migrations
│   │   └── schemas/                # Pydantic request/response models
│   └── requirements.txt
├── frontend/
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── img/
│   └── templates/                  # Dashboard, Detection, Inventory, Alerts, Analytics, Settings pages
├── models/
│   ├── pretrained/                 # Baseline COCO-pretrained YOLO weights
│   └── custom/                     # Custom-trained project model weights
├── datasets/
│   ├── raw/                        # Original captured images
│   ├── annotated/                  # YOLO-format labeled dataset
│   └── v1/, v2/...                 # Versioned dataset snapshots
├── database/
│   └── smartshelf.db               # SQLite database file (generated at runtime)
├── configuration/
│   └── settings.example.json       # Default/example configuration
├── tests/
│   ├── test_detection.py
│   ├── test_stock_logic.py
│   ├── test_api.py
│   └── fixtures/                   # Sample test images/videos
├── scripts/
│   ├── train_model.py              # Custom YOLO training script
│   ├── evaluate_model.py           # Evaluation metrics script
│   └── seed_demo_data.py           # Populate DB with demo scenario data
├── docs/
│   ├── PROJECT_PRD.md              # This document
│   └── demo_script.md
└── README.md
```

---

# 21. Development Phases

| Phase | Goal | Deliverables | Dependencies | Acceptance Criteria |
|---|---|---|---|---|
| **Phase 0 — Requirements** | Finalize scope | This PRD (approved) | None | PRD reviewed and accepted by team/guide |
| **Phase 1 — Project Setup** | Scaffold repo, environments | Folder structure, `requirements.txt`, virtualenv, FastAPI "hello world" | Phase 0 | Backend boots and serves a health-check endpoint |
| **Phase 2 — Dataset** | Collect & annotate custom dataset | Raw images, YOLO-format annotations, dataset split | Phase 1 | Dataset validated (correct format, class balance checked) |
| **Phase 3 — YOLO Integration** | Load pretrained model, run baseline inference | Working inference script on sample images | Phase 1 | Pretrained model produces bounding boxes on a test image |
| **Phase 4 — Detection Pipeline** | Wrap inference in a reusable service (FR-01, FR-02) | `cv/` module with detect() function | Phase 3 | Unit test passes on fixture images |
| **Phase 5 — Counting & Multi-Product** | Implement counting logic (FR-03, FR-08) | Counting module + tests | Phase 4 | Counts match manual ground truth on fixtures |
| **Phase 6 — Stock Logic** | Implement ROI + threshold logic (FR-04–07, Sections 13–14) | `logic/` module + tests | Phase 5 | Staged scenarios classify correctly |
| **Phase 7 — Database** | Implement schema (Section 17) | SQLAlchemy models, migrations | Phase 1 | Tables created; sample inserts succeed |
| **Phase 8 — Backend/API** | Implement all endpoints (Section 18) | Working FastAPI routes | Phases 4–7 | All endpoints return correct responses in manual/API testing |
| **Phase 9 — Dashboard** | Build frontend pages (Section 19) | HTML/CSS/JS pages wired to API | Phase 8 | All Section 15 components render with real data |
| **Phase 10 — Alerts** | Implement alert generation + UI (Section 16) | Alert logic + toast/panel UI | Phases 6, 8, 9 | Staged low-stock/empty scenario produces visible alert |
| **Phase 11 — Model Training** | Train custom model on collected dataset | Trained `.pt` weights file | Phase 2 | Model loads and runs inference successfully |
| **Phase 12 — Testing** | Full test pass (Section 22) | Test suite + results log | Phases 4–11 | All defined test cases pass or issues documented |
| **Phase 13 — Optimization & Evaluation** | Tune thresholds, measure metrics (Section 23) | Evaluation report | Phase 12 | Precision/recall/mAP/FPS numbers recorded |
| **Phase 14 — Final Documentation & Demo Prep** | Prepare report, demo script, polish UI | Final report, demo script, rehearsed demo | All prior phases | Demo (Section 27) runs end-to-end without failure |

---

# 22. Testing Strategy

- **Unit Testing:** Test individual functions — stock threshold classification, ROI-detection association, NMS wrapper, confidence filtering. (`pytest`)
- **Integration Testing:** Test the full pipeline from image input → detection → stock status → DB write → API response.
- **AI Model Testing:** Run the model against the held-out test split; confirm outputs are structurally valid (boxes within image bounds, valid class IDs).
- **API Testing:** Test each endpoint in Section 18 for correct status codes and payload shapes, including error paths (e.g., uploading a non-image file).
- **UI Testing:** Manual click-through of every page/component listed in Section 19 to confirm no broken rendering or dead buttons.
- **Performance Testing:** Measure end-to-end latency for image mode and FPS for live mode on the actual demo laptop.
- **Failure Testing:** Deliberately trigger each condition in Section 25 (unplug camera, corrupt a video file, delete the model file, stop the DB file's access) and confirm graceful handling.
- **Accuracy Evaluation:** Run Section 23's metrics on the test split after training.

**Measurable acceptance criteria:** A test case passes if its documented expected output (exact match for logic tests, "no crash + clear message" for failure tests, or "metric computed and recorded" for model tests) is observed.

---

# 23. AI Model Evaluation

Metrics to compute on the held-out test split after training the custom model (Section 12):

- **Precision** — proportion of predicted boxes that are correct detections.
- **Recall** — proportion of actual products that were successfully detected.
- **F1 Score** — harmonic mean of precision and recall.
- **mAP (mean Average Precision)** — standard object-detection metric, typically reported at IoU=0.5 (mAP@0.5) and across IoU=0.5:0.95.
- **Inference Time** — average milliseconds per frame on the demo laptop's CPU (and GPU, if used).
- **FPS** — frames processed per second in live mode, derived from inference time plus pipeline overhead.

**Reporting rule (explicit, non-negotiable per the brief):** None of these numbers are to be stated in the final report or dashboard until they have actually been computed by running the evaluation script against real test data. No placeholder or assumed accuracy value (e.g., "95% accurate") may be used.

---

# 24. Security

Reasonable, non-over-engineered controls appropriate for a student software project:

- Basic username/password authentication gating the Settings/Admin area (hashed password storage, e.g., via `passlib`/bcrypt — not plaintext).
- Input validation on all file uploads (file type/size checks) to avoid processing arbitrary or oversized files.
- Parameterized database queries (via SQLAlchemy ORM) to avoid SQL injection.
- The application is intended to run on `localhost` for the demo; no requirement to secure it against public internet exposure.
- No requirement for encryption-at-rest of the SQLite file, given the non-sensitive nature of the demo data.

---

# 25. Error Handling

| Condition | Expected Behavior |
|---|---|
| Missing model file | Startup/API call returns a clear "Model not found" error; dashboard shows a Model Failure alert; no crash. |
| Invalid image (corrupt/unsupported format) | Upload endpoint returns 400 with a clear message; no server crash. |
| Corrupt video file | Video processing catches the decode error, returns a clear message, and marks the run as failed rather than hanging. |
| Camera unavailable | Live-mode start attempt returns a "camera unavailable" status; dashboard shows Camera/Input Failure indicator; system remains usable for image/video modes. |
| Database unavailable/locked | API returns a 500 with a generic "storage error" message; in-memory operation (e.g., latest detection display) can continue where feasible, but is documented as a known limitation. |
| No detections in a frame | System reports zero counts and (per ROI logic) an EMPTY or "no data" status rather than treating it as an error. |
| Low-confidence detections | Detections below the configured confidence threshold are silently filtered out (not shown, not counted) — this is expected behavior, not an error. |

---

# 26. Deployment

**Target:** Single Windows 10/11 laptop, fully local, no cloud dependency after initial setup.

1. **Python environment:** Install Python 3.10+; create a virtual environment (`python -m venv venv`); activate it.
2. **Dependencies:** `pip install -r backend/requirements.txt` (installs FastAPI, Uvicorn, Ultralytics, OpenCV, NumPy, SQLAlchemy, etc.).
3. **Model files:** Place the pretrained baseline weights in `models/pretrained/` and (after training) the custom weights in `models/custom/`; set the active model path in `configuration/settings.example.json` or via the Settings page.
4. **Database:** SQLite file is auto-created at `database/smartshelf.db` on first run (via SQLAlchemy's `create_all` or a lightweight migration script); no separate DB server install needed.
5. **Starting backend:** `uvicorn backend.app.main:app --reload` (or a packaged `run.py` script) starts the API server on `localhost:8000`.
6. **Starting frontend:** Served directly by FastAPI via Jinja2 templates/static files at the same address (`localhost:8000`), so no separate frontend server/build step is required — simplifies demo setup.
7. **Local usage:** Open a browser to `http://localhost:8000` to access the dashboard; webcam access is handled by OpenCV directly from the backend process.

No cloud deployment is required or recommended for this academic project; it is explicitly out of scope (Section 3.2), with a note in Future Enhancements (Section 30) for anyone wishing to extend it later.

---

# 27. Demo Scenario (5–10 minutes)

1. Start the application (`uvicorn` command or a double-clickable `run.bat`) — dashboard loads at `localhost:8000`.
2. Open the Dashboard — show empty/initial state and explain the layout.
3. Go to the Detection page and upload a pre-staged image of a fully stocked shelf.
4. Show the bounding boxes and class labels drawn on the annotated image.
5. Show the resulting product counts (total and per class).
6. Show the shelf's stock status badge as AVAILABLE (green).
7. Upload a second staged image showing a partially emptied shelf; show the status change to LOW STOCK (amber) and the resulting alert appearing in the Alerts panel.
8. Upload a third staged image showing a fully empty shelf; show the status change to EMPTY (red) and a second alert firing.
9. Switch to the Inventory page to show the shelf's history across these three runs.
10. Switch to the Analytics page to show the trend chart reflecting the three states.
11. (Optional, if a webcam is available) Briefly demonstrate Live Mode by holding a real product in front of the camera.
12. Close by opening the Model Information page and explaining the pretrained-vs-custom-model distinction and the measured evaluation metrics (Section 23).

---

# 28. Expected Outcomes

Adapted from the PPT's Chapter 5 outcomes to the actual software-only implementation:

- The system will perform real-time-capable product detection on shelf images/video/webcam feed using a YOLO model, with measured (not assumed) accuracy reported at project completion.
- The system will effectively distinguish AVAILABLE, LOW STOCK, and EMPTY conditions per configured shelf region, based on transparent, configurable threshold logic.
- The system will generate automatic, visible alerts when stock is low or a shelf is empty, supporting faster response than manual shelf-checking.
- A user-friendly, web-based dashboard will display live stock status, recent detections, and historical trends.
- The system will demonstrably reduce the manual effort needed to assess shelf stock state during a demo scenario, and will support better-informed restocking decisions in a simulated retail context — entirely through software running on a normal Windows laptop, with no IoT hardware, sensors, or Raspberry Pi involved.

---

# 29. Limitations

- **Camera angle:** Detection accuracy depends heavily on a consistent, reasonably front-facing camera angle relative to the shelf; extreme angles were not designed for.
- **Occlusion:** Products partially blocked by other objects, hands, or carts may be missed or undercounted.
- **Lighting:** Poor or highly variable lighting can reduce detection confidence and increase false negatives.
- **Similar-looking products:** Products with near-identical packaging/color may be confused by the model, especially with a small training dataset.
- **Custom dataset dependency:** Real classification accuracy is only meaningful for the specific product classes the team actually collects and trains on; the system does not generalize to arbitrary, unseen retail products without retraining.
- **Detection accuracy:** No detection system is perfect; expect some false positives/negatives, quantified in Section 23's evaluation, not assumed to be zero.
- **Empty-shelf interpretation:** "Empty" is defined relative to a manually configured expected capacity and ROI, not an innate scene understanding — misconfiguration will produce misleading status.
- **Hardware-independent limitations:** Running on CPU-only hardware limits achievable live-mode FPS compared to a GPU-equipped setup; this is disclosed rather than hidden.

---

# 30. Future Enhancements

- Multi-camera support (multiple simultaneous shelf feeds)
- Cloud deployment for remote multi-site monitoring
- Edge deployment (e.g., actually running on a Raspberry Pi or similar device, if the team later wants to revisit that original hardware idea)
- Advanced multi-object tracking across frames (e.g., DeepSORT) to reduce double-counting in live video
- Automated restocking prediction based on consumption trends
- Demand forecasting using historical detection data
- Mobile companion application for alerts on the go
- Integration with a real POS/ERP system for automated restock ordering
- Email/SMS/push notification channels for alerts

---

# 31. Academic Mapping

To satisfy academic reporting expectations, this implementation maps back onto the PPT's chapter structure as follows:

- **Existing System (per PPT Chapter 2.2, reframed for this domain):** Manual or basic camera-based shelf monitoring with no automated classification, counting, or alerting — non-interactive, hardware-intensive if sensor-based, no predictive/analytic capability.
- **Proposed System (this project):** A software-only YOLO-based product detection and stock-monitoring pipeline with a real-time dashboard, automated stock classification, and alerting — no hardware installation required beyond an optional standard webcam.
- **Advantages:** Low-cost (no sensors/Raspberry Pi), real-time-capable, runs on hardware students already own, scalable to more product classes/shelves via configuration, reduces manual shelf-checking effort.
- **Functional Requirements:** Section 7 (FR-01 to FR-18), derived directly from PPT Chapter 4.1, expanded with implementation-necessary detail.
- **Non-Functional Requirements:** Section 8, derived directly from PPT Chapter 4.2, with realistic targets substituted for the PPT's qualitative statements.
- **Expected Outcomes:** Section 28, derived directly from PPT Chapter 5, translated into the software-only scope.

---

# 32. Acceptance Criteria (Project-Level Checklist)

The project is considered complete only when **all** of the following are true:

- [ ] Detection pipeline runs on image, video, and live webcam input (FR-09–11)
- [ ] Products are detected and classified into the team's configured custom classes using a custom-trained model (Section 11-B, 12)
- [ ] Product counting works per frame and per shelf region (FR-03–04)
- [ ] Stock status (AVAILABLE/LOW STOCK/EMPTY) is computed via configurable thresholds and verified on staged scenarios (FR-05–07, Section 13)
- [ ] Dashboard displays all components listed in Section 15 with real data
- [ ] Alerts fire correctly for low-stock and empty-shelf transitions and are visible in-app (FR-13, Section 16)
- [ ] Detection history and basic analytics are functional (FR-14–15)
- [ ] Configuration of shelves/thresholds/product classes/model path works without code changes (FR-16–17)
- [ ] All Section 25 error conditions have been manually tested and handled gracefully (FR-18)
- [ ] Model evaluation metrics (Section 23) have been computed and are ready to present — no invented numbers
- [ ] The full demo scenario (Section 27) has been rehearsed end-to-end in under 10 minutes
- [ ] No IoT sensors, Raspberry Pi, or physical hardware automation exist anywhere in the delivered system
- [ ] The PPT inconsistency (Section 2.3) is documented in the final project report, not hidden

---

# 33. Risks and Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Small/insufficient custom dataset leads to poor model accuracy | Weak demo results, low evaluation scores | Start dataset collection early (Phase 2); use data augmentation; keep the number of product classes small (3–8) to make each class well-represented |
| Live webcam performance too slow on CPU-only laptops | Choppy demo, poor "real-time" impression | Use YOLOv8n (nano) variant; reduce inference resolution; fall back to image/video-mode demo as primary if live mode is too slow |
| Team under-estimates annotation effort | Delays to Phase 2/11 | Use Roboflow's semi-automated annotation and augmentation tools; scope down to fewer classes if needed |
| Misconfigured shelf ROI/capacity gives misleading stock status during demo | Embarrassing incorrect result live in front of evaluator | Rehearse and lock in ROI/threshold configuration well before demo day; test with the exact staged images/props to be used in the demo |
| Evaluator questions the Indoor Wellness vs. Product Detection discrepancy | Perceived lack of rigor if unaddressed | Proactively present Section 2.3's inconsistency analysis in the report/demo introduction, framing it as a deliberate, documented scoping decision |
| SQLite contention during simultaneous live-mode writes and dashboard reads | Occasional UI lag or minor errors | Use short-lived DB sessions, WAL mode for SQLite, and keep write volume low (batched per analysis run, not per frame) |

---

# 34. Open Questions

The following could not be resolved from the PPT and require a team decision (a sensible default has been assumed in each case per Section 2.4, so these are not blockers, but the team should confirm):

1. **Exact product classes to use for the custom dataset** — the PPT never names specific products. *(Default assumed: team selects 3–8 physically available items.)*
2. **Exact numeric values for `expected_capacity` per shelf and for the low-stock/empty thresholds** — the PPT gives no numbers. *(Default assumed: Section 13's suggested values, configurable.)*
3. **Whether a GPU will be available for training/inference** — affects achievable FPS and training time, not stated anywhere in the PPT. *(Default assumed: CPU-only baseline, with GPU as an optional speed-up.)*
4. **Whether the team wants to retain the "Indoor Wellness" project title on the cover page of the final report, or officially rename it to reflect the actual implementation** — this is an administrative/academic decision for the team and guide, not resolvable from the PPT content alone.

---

*End of PROJECT_PRD.md*
