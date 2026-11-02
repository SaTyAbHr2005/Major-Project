# Module 2: Researcher & Admin Application

**Project:** Secure and Privacy-Preserving Federated Deep Learning Training Platform for Medical Imaging
**Owner:** Member 4 (Frontend + Desktop + Mobile + Communication)
**Location:** `frontend/` (one web app; views in `frontend/src/views/`). Desktop packaging is Module 21 and not done yet.
**Status:** Implemented, with some screens still mock (marked with a banner)
**Last Updated:** 2026-11-02

---

## Status & Implementation Checklist

- [x] Sign-in for Admin and Researcher through Module 1 (no offline or demo login)
- [x] Researcher access request (registers the account as pending)
- [x] Admin: dashboard, researcher approvals, disease reports, hospitals, federation jobs and rounds, global models, stored models, update validation, users, audit log
- [x] Researcher: dashboard, training requests, live monitoring, hospitals, profile and settings
- [x] Federation job form: strategy, differential privacy and Module 12 security settings
- [x] Live Flower panel: round progress, each hospital's step, start, stop and aggregate now
- [x] Global model evaluation with progress, and promotion to MAIN
- [x] State management with Zustand stores in `src/stores/`; demo data isolated in `src/mocks/`
- [ ] Experiments screen (mock; Module 20)
- [ ] Notifications and Messages screens (mock; Module 18)
- [ ] Halt, Quarantine and Export dialogs (mock)
- [ ] Per-hospital hardware telemetry (Module 13) and model export (Module 15)

---

## 1. Module Purpose & Core Architecture

Module 2 is the central application for the Admin and Researcher roles. Admins run federated training across hospitals; Researchers publish training requests and follow hospitals' local training. Neither role accesses raw patient data.

### Privacy Rule
> Raw medical images do not flow to or through Module 2. It shows metadata, aggregate training progress, update verdicts and global model evaluations only.

```text
                               +------------------------------------------+
                               |   Module 2: Admin and Researcher screens |
                               |   (React + Vite, part of frontend/)      |
                               +--------------------+---------------------+
                                                    |
                         +--------------------------+--------------------------+
                         |                                                     |
                         v                                                     v
          +-----------------------------+                       +-----------------------------+
          |  Researcher                 |                       |  Admin                      |
          +-----------------------------+                       +-----------------------------+
          | - Publish training requests |                       | - Approve users             |
          | - Follow local progress     |                       | - Create and run jobs       |
          | - View participating        |                       | - Approve hospital joins    |
          |   hospitals                 |                       | - Evaluate, promote models  |
          |                             |                       | - Update validation, audit  |
          +--------------+--------------+                       +--------------+--------------+
                         |                                                     |
                         +--------------------------+--------------------------+
                                                    |
                                                    v
                               +------------------------------------------+
                               |   Backend (Node.js, REST + Socket.io)    |
                               |  - Module 1: auth, roles, audit          |
                               |  - Module 9: jobs, rounds, Flower server |
                               |  - Modules 10, 12: DP and update checks  |
                               +------------------------------------------+
```

---

## 2. Responsibilities Matrix

| Functional Area | Handled by Module 2 | Related Modules |
|---|:---:|:---:|
| Federation job setup, start, stop and aggregation controls | ✅ | Module 9 |
| Differential privacy settings and per-job budget view | ✅ | Module 10 |
| Module 12 settings, update verdicts and rejection reasons | ✅ | Module 12 |
| Hospital list with request and round participation | ✅ | Modules 1, 9 |
| Global model versions, evaluation and promotion | ✅ | Modules 9, 15 |
| Researcher approval and user management | ✅ | Module 1 |
| Audit log view | ✅ | Modules 1, 14 |
| Per-hospital hardware telemetry | ❌ *(not implemented; the screen says so)* | Module 13 |
| Messaging and Admin/Researcher notifications | ❌ *(mock)* | Module 18 |
| Raw image loading | ❌ *(not allowed)* | Modules 3, 4 |
| Local model training | ❌ | Modules 7, 9 |

---

## 3. Role-Based Access Control (RBAC) in Module 2

Module 2 follows the roles defined by Module 1:

```javascript
ROLES.RESEARCHER = 'researcher'
ROLES.ADMIN      = 'admin'
```

Hospital Operators are a third, separate role with their own screens (Module 3).

### Role Capabilities in Interface

| Capability | Researcher | Admin |
|---|:---:|:---:|
| Publish training requests and follow their progress live | ✅ | ❌ |
| View hospitals | ✅ (request participation) | ✅ (also registry and round participation) |
| Create and run federation jobs, approve hospital joins | ❌ | ✅ |
| Evaluate and promote global models | ❌ | ✅ |
| View update validation | ❌ | ✅ |
| View audit logs | ❌ | ✅ |
| Approve, reject or suspend users | ❌ | ✅ |
| Export model weights | ❌ | ❌ (not implemented) |

---

## 4. Screen-by-Screen Architecture

### 1. Dashboard (`DashboardView.jsx`)
- **Admin:** active hospitals, federation jobs, global models and training requests, from real data.
- **Researcher:** own training requests, participating hospitals, hospitals training now and local training finished.

### 2. Federation Jobs & Rounds (`FederatedTrainingView.jsx`, admin)
- Create a Module 9 job (task, architecture, class mapping, minimum participants, round plan, strategy, DP and Module 12 settings), delete or archive jobs and rounds. Jobs appear under In progress, Completed and Archived.
- Review hospital join requests with each hospital's local eligibility summary, then approve or reject them.
- **Start federation** runs the job through one Flower server, with a live panel of round status and each hospital's step. **Stop** ends the server, and **Aggregate now** closes a round early once enough updates have been sent.
- Manual rounds: **Open round N** takes a deadline (required, in the future) and minimum participants. Module 9 chooses the base model: the job's latest global model, or one shared freshly initialised seed (`SEED_<job_id>`) when the job has none yet. Hospitals download it (`GET /federation/rounds/:round_id/base-model/:file`) and verify its checksum before training. Only one round per job can be in progress.
- Per-round participants (hospital name and id) and their update status, including FLAGGED and QUARANTINED badges, plus the DP budget panel for DP jobs.
- **Aggregate Round** opens a confirmation listing every hospital that joined the round, by name and id, with its update status; before the deadline it warns that the round closes early (`force_close`, Module 9 still requires the minimum accepted updates). It then shows the aggregation in progress and the result. Every active hospital that did not join the round gets a `ROUND_CLOSED` notification; opening a round sends `ROUND_OPEN` to every active hospital.

### 3. Training Requests (`ResearcherTrainingView.jsx`, researcher)
- Publish a training request (disease, task, description, architecture from the Module 6 list, epochs, batch size) and follow each hospital live through Socket.io room events.
- A training request is separate from a federation job: it does not create one.

### 4. Monitoring (`MonitoringView.jsx`, researcher)
- Live per-hospital epoch, loss and accuracy for the researcher's open requests.

### 5. Hospitals (`HospitalsView.jsx`)
- Training-request participation for both roles. Admins also see the hospital registry and federation round participation.

### 6. Global Models (`AdminFederationModelsView.jsx`, `ModelsView.jsx`, admin)
- Module 9 global model versions with evaluations, stored-model records and the Module 6 architecture catalog.
- Evaluate a version on the held-out set with live progress, and promote it to MAIN. These endpoints are admin-only, so researchers see a notice instead.

### 7. Researcher Approvals and Users (`AdminApprovalView.jsx`, `UserManagementView.jsx`, admin)
- Approve or reject pending researchers, and approve, reject or suspend any account.

### 8. Disease Reports (`AdminDiseaseModelsView.jsx`, admin)
- Model and training information per disease, from `/admin/diseases` and `/admin/reports`.

### 9. Update Validation (`SecurityView.jsx`, admin)
- Every hospital update with the server's verdict and rejection reason, with counts of accepted, rejected or quarantined, and expired or failed updates.
- Its status list still calls Module 11 "Pending"; this text should be updated when Module 11 closes.

### 10. Audit Logs (`AuditLogsView.jsx`, admin)
- The Module 1 audit trail: logins, approvals, training-request participation and progress. Admin federation actions are not recorded yet.

### 11. Profile and Settings (`ProfileView.jsx`, `SettingsView.jsx`)
- The signed-in account, password change and sign-out; researchers also see the requests they published.

### 12. Sign-in and Access Request (`LoginView.jsx`, `AdminLoginView.jsx`, `RequestAccessView.jsx`)
- Sign-in through Module 1. Researchers register through the access request form and stay pending until an Admin approves.

### 13. Mock Screens (`ExperimentsView.jsx`, `NotificationsView.jsx`, `MessagesView.jsx`, `StatesDemoView.jsx`)
- Demo data with a "Mock data" banner, until Modules 18 and 20 provide real data.

---

## 5. Dialogs

1. **Aggregate Round (`components/federation/AggregateRoundDialog.jsx`):** real; see screen 2.
2. **Halt Training, Quarantine and Export Weights (`components/modals/`):** mock with a banner. They do not halt training, quarantine a hospital or export weights. Module 12 quarantines an update automatically at intake, and no action exists to release it.

---

## 6. How to Run Locally

### Prerequisites
- Node.js >= 20.19 (the backend's mongoose 9 requires it)
- npm >= 9.0.0

The researcher desktop is part of the single consolidated app in `frontend/` (the separate
`frontend/researcher-desktop/` folder was removed). See the root `README.md` for the full setup.

### Installation & Execution
```powershell
# 1. From the repository root, go to the consolidated frontend app
cd frontend

# 2. Install dependencies (if not already installed)
npm install

# 3. Start development server
npm run dev
```

The application runs at **`http://localhost:5173`** (or next available port), or at `https://localhost:5173` in TLS mode (`VITE_API_URL` is https and `certs/` exists).

### Environment Configuration
The application reads its backend API endpoint from `.env`:
```env
VITE_API_URL=http://localhost:5000/api/v1
```
Sign-in requires the backend: there is no offline or demo login. If the backend is unreachable, sign-in fails with an error.

### Data sources per screen
| Screen | Role | Source |
|---|---|---|
| Dashboard | admin | M9 jobs/rounds/global models (`/federation/*`), `/admin/telemetry`, `/training-requests` |
| Dashboard | researcher | own training requests (`/training-requests/my`) |
| Training (researcher) | researcher | `POST /training-requests`, `/training-requests/my`, live Socket.io room events |
| Monitoring | researcher | own open requests' per-hospital epoch/loss/accuracy, live |
| Hospitals | both | `/training-requests` participation; admins also `/admin/telemetry` and M9 rounds |
| Federation Jobs & Rounds | admin | `/federation/jobs`, `POST /federation/jobs`, `POST /federation/rounds`, `POST /federation/rounds/aggregate`, `/federation/jobs/:id/flower/*`, participant decisions |
| Global Models | admin | `/federation/models` (+ evaluations), `/admin/stored-models`, `/federation/architectures` |
| Update Validation (Security) | admin | M9 job details: every submitted update and its rejection reason |
| User Management | admin | `GET /admin/users`, `POST /admin/users/:id/{approve,reject,suspend}` |
| Audit Log | admin | `GET /admin/audit-logs` |
| Profile / Settings | both | `/auth/me`, `POST /auth/change-password` |
| Experiments, Notifications, Messages, States demo | - | **Mock data** (Modules 18/20 not finished), shown with a "Mock data" banner |

Hardware/GPU telemetry per hospital (Module 13) and model export (Module 15) are not implemented; the screens say so instead of showing figures. Differential privacy (Module 10) and Byzantine detection (Module 12) are implemented: the admin job form sets them, and the federation and Security screens show the results. The Quarantine and Export dialogs are still mock.
