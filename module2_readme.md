# Module 2: Researcher & Consortium Admin Desktop Application

**Project:** Secure and Privacy-Preserving Federated Deep Learning Training Platform for Medical Imaging  
**Owner:** Member 4 (Frontend + Desktop + Mobile + Communication)  
**Location:** `frontend/` (consolidated app; researcher views in `frontend/src/views/`)  
**Status:** Functionally Complete & Verified (`[x] Completed`)  
**Last Updated:** 2026-09-23

---

## Status & Implementation Checklist

- [x] Consortium Lead Investigator Dashboard with live metrics
- [x] Federated Training Session Management & Round Progression
- [x] Experiments & Hyperparameter Tuning Manager
- [x] Hospital Node Registry & Quorum Tracking (8/8 Ring Nodes)
- [x] Model Registry with Comparative Architectural Matrix & Radar Chart
- [x] Security Center with Byzantine Vector Space 2D Projection & Privacy Budget Circular Gauge
- [x] Cryptographic Merkle Root Ledger & Audit Log Payload Inspector
- [x] Alert Center with Byzantine Fault Clamping
- [x] Secure Dispatch & Kyoto Hospital Chat Communication Channel
- [x] Investigator Profile & Consortium Governance Identity
- [x] Settings with Hard-Locked Zero-Raw-Data Enclave Enforcement
- [x] Interactive States Demo (Normal, Loading, Empty, Error, Offline)
- [x] Halt Training Confirmation Modal with Egress Revocation
- [x] Byzantine Node Quarantine Dialog with Cryptographic Exclusion
- [x] Model Weights Export Modal with Differential Privacy Certificate
- [x] Role Switch Modal (Lead FL Investigator ↔ Consortium Admin Tier-1)
- [x] MedFL.Secure Hardware Attestation Login Screen (FIDO2 / YubiKey / SAML 2.0)
- [x] Consortium Access Request Wizard (`SEC-792-REQ`)
- [x] State Management Architecture (`AppContext.jsx`) with dynamic metrics
- [x] Automated Module 1 Backend API Integration (`.env` configuration)
- [x] Documentation & Production Verification Complete

---

## 1. Module Purpose & Core Architecture

Module 2 is the primary orchestration desktop application for clinical researchers, principal investigators, and consortium administrators. It provides a central command center for managing federated medical imaging training rounds across distributed hospital nodes without ever centralizing or accessing raw patient data.

### Cardinal Privacy Invariant
> **ZERO-RAW-DATA ARCHITECTURAL ENFORCEMENT**  
> Under no circumstances does raw medical imaging data (DICOM series, NIfTI volumes, EHR files) flow to or through Module 2. The central researcher workstation orchestrates training sessions, monitors differential privacy budgets ($\epsilon, \delta$), inspects gradient update distributions, and approves aggregated global models exclusively.

```text
                               +------------------------------------------+
                               |     Module 2: Researcher Desktop         |
                               | (React 19 + Vite + CSS Design System)    |
                               +--------------------+---------------------+
                                                    |
                         +--------------------------+--------------------------+
                         |                                                     |
                         v                                                     v
          +-----------------------------+                       +-----------------------------+
          |  Lead Investigator Workflows|                       | Consortium Admin Operations |
          +-----------------------------+                       +-----------------------------+
          | • Hyperparameter Config     |                       | • Node Quarantine / Ejection|
          | • Quorum Monitoring         |                       | • User & Access Approval    |
          | • Convergence Curves        |                       | • Byzantine Vector Clamp    |
          | • Differential Privacy View |                       | • Cryptographic Merkle Audit|
          | • Model Export & Packaging  |                       | • Global Key Revocation     |
          +--------------+--------------+                       +--------------+--------------+
                         |                                                     |
                         +--------------------------+--------------------------+
                                                    |
                                                    v
                               +------------------------------------------+
                               |   Backend & Security Infrastructure      |
                               |  • Module 1: Auth & RBAC (/api/v1/auth)  |
                               |  • Module 9: Flower FL Engine Orchestrator|
                               |  • Module 12: Byzantine Defense Vector   |
                               |  • Module 13/14: Telemetry & Merkle Logs |
                               +------------------------------------------+
```

---

## 2. Responsibilities Matrix

| Functional Area | Handled by Module 2 | Related Modules |
|---|:---:|:---:|
| Federated round orchestration & start/pause/halt controls | ✅ | Module 9 (FL Engine) |
| Differential privacy expenditure tracking ($\epsilon=1.24 / 1.50$) | ✅ | Module 10 (Differential Privacy) |
| Byzantine node anomaly detection & 2D gradient projection | ✅ | Module 12 (Byzantine Defense) |
| Hospital node quorum & telemetry monitoring (8 clinical nodes) | ✅ | Module 13 (Telemetry) |
| Merkle root ledger & cryptographic audit record inspection | ✅ | Module 14 (Audit Storage) |
| Comparative architectural matrix & 5-axis radar evaluation | ✅ | Module 6, 15 (Model Versioning) |
| Consortium access request wizard (`SEC-792-REQ`) | ✅ | Module 1 (Auth & Approval) |
| Secure inter-hospital dispatch & operator communication | ✅ | Module 18 (Communication) |
| Enforcing strict client-side Zero-Raw-Data Lock | ✅ | All modules |
| Raw image loading / local DICOM storage | ❌ *(Strictly prohibited)* | Handled strictly inside Module 3/4 |
| Local model gradient computation | ❌ | Handled inside Module 7/9 |

---

## 3. Role-Based Access Control (RBAC) in Module 2

Module 2 strictly honors the role models defined by Module 1:

```javascript
ROLES.RESEARCHER = 'researcher'      // Lead FL Investigator
ROLES.ADMIN      = 'admin'           // Consortium Admin Tier-1
```

### Role Capabilities in Interface

| Capability | Lead FL Investigator (`researcher`) | Consortium Admin Tier-1 (`admin`) |
|---|:---:|:---:|
| View Training Dashboard & Convergence Curves | ✅ | ✅ |
| Configure Experiments & Hyperparameters | ✅ | ✅ |
| Inspect Model Registry & Architectures | ✅ | ✅ |
| View Hospital Node Status & Latency | ✅ | ✅ |
| View Merkle Audit Trail | ✅ (Read-Only) | ✅ (Full Verification) |
| Quarantine / Exclude Byzantine Hospital Node | ❌ *(Requires Admin Approval)* | ✅ *(Instant Exclusion)* |
| Emergency Halt Consortium Training | ❌ *(Can Request Halt)* | ✅ *(Instant Global Halt)* |
| Approve New Researcher Access Requests | ❌ | ✅ |
| Export Global Model Weights Bundle | ✅ (DP-Attested) | ✅ (Signed by Enclave) |

---

## 4. Screen-by-Screen Architecture

Module 2 contains 16 distinct operational screens and dialog suites:

### 1. Consortium Dashboard (`DashboardView.jsx`)
- **Federated Quorum Strip:** Real-time indicator of active participating clinical nodes (8/8 online, FedAvg v3.2, 100% quorum).
- **Executive Metric Cards:** Active Experiments (4), Global Aggregation Rounds (42/50), Differential Privacy Budget Spent ($\epsilon=1.24$, 82.6%), Global Validation ROC-AUC (0.942).
- **Convergence Curves:** Live multi-metric training progress charting loss decay and validation accuracy across rounds.
- **Node Latency & Performance Stack Bar:** Inter-hospital synchronization times across Tokyo, Berlin, Boston, and Geneva nodes.

### 2. Federated Training Center (`FederatedTrainingView.jsx`)
- Round-by-round training controller with active status indicators (`RUNNING`, `PAUSED`, `STOPPED`).
- Aggregation algorithm selector (Federated Averaging `FedAvg`, `FedProx`, `Scaffold`).
- Round countdown, participant readiness gates, and global parameter aggregation stream.
- One-click trigger for the **Halt Training Dialog**.

### 3. Experiments & Tuning (`ExperimentsView.jsx`)
- Experiment registry tracking ID, modality (Pneumonia Chest X-Ray, Brain MRI), target architecture, client cohort, and convergence status.
- Hyperparameter tuning drawer: learning rate, momentum, weight decay, local epochs, batch size, and clipping norm ($C=1.0$).

### 4. Hospital Node Registry (`HospitalsView.jsx`)
- Comprehensive card and tabular registry of all consortium hospital nodes (e.g., St. Jude Children's, Charité Berlin, Johns Hopkins, Kyoto University Hospital).
- Node attestation status (Intel SGX2 FIPS Validated), compute tier, local sample counts, and round contribution history.
- Contextual actions: Direct Dispatch Message, Telemetry Inspector, and Quarantine Trigger.

### 5. Model Registry (`ModelsView.jsx`)
- Catalog of consortium-approved architectures: ResNet-18, EfficientNet-B0, DenseNet-121, MobileNetV3-Large.
- **Comparative Architectural Matrix:** Tabular comparison of parameter count, MACs/FLOPs, memory footprint, and convergence rates.
- **5-Axis Radar Chart:** Multi-dimensional visualization rating Accuracy, Memory Efficiency, Throughput, Privacy Resilience, and Generalizability.

### 6. Security Center & Byzantine Defense (`SecurityView.jsx`)
- **Byzantine Vector Space 2D Projection:** Dynamic scatter visualization projecting hospital gradient updates onto a 2-dimensional principal component plane, isolating malicious/outlier gradient vectors (e.g., Kyoto Hospital anomaly).
- **Privacy Budget Circular Gauge:** SVG radial gauge visualizing cumulative differential privacy expenditure ($\epsilon=1.24 / 1.50$, $\delta=10^{-5}$) with safety color thresholds.
- **Intrusion Defense & Tamper Log:** Real-time log of gradient clipping triggers, poisoned weight rejections, and TLS handshake anomalies.

### 7. Audit Logs & Merkle Inspector (`AuditLogsView.jsx`)
- Cryptographic event ledger with SHA-256 state hashes and Merkle root checkpoints.
- **Cryptographic Payload Inspector:** Embedded syntax-highlighted JSON viewer allowing researchers to inspect audit event signatures, public keys, and nonces.

### 8. Notifications & Alert Center (`NotificationsView.jsx`)
- Multi-tier alert stream categorized by urgency (`CRITICAL`, `WARNING`, `INFO`).
- Interactive Byzantine fault clamp actions and automated quorum warning dismissals.

### 9. Secure Messages & Dispatch (`MessagesView.jsx`)
- End-to-end encrypted dispatch channels between consortium lead investigators and hospital clinical operators.
- Hardware telemetry drawer embedded inside chat (showing remote node GPU temp, VRAM, and daemon health).

### 10. Profile & Governance Identity (`ProfileView.jsx`)
- Verified clinical investigator identity: Dr. Elena Rostova, M.D., Ph.D.
- Affiliation credentials (Stanford Medicine AI Lab), ORCID, NPI, YubiKey 5C FIPS token ID, and Intel SGX enclave attestation fingerprints.

### 11. Settings & Platform Constraints (`SettingsView.jsx`)
- Telemetry refresh intervals, theme configurations, and API endpoint overrides.
- **Hard-Locked Zero-Raw-Data Toggle:** Permanently locked visual switch enforcing zero medical image ingress with explanatory policy text.

### 12. States & Dialogs Showcase (`StatesDemoView.jsx`)
- Interactive tester allowing reviewers to instantly toggle global application states: `Normal`, `Loading Skeleton`, `Empty / No Data`, and `Error / Disconnected`.

### 13. Authentication & Enclave Login (`LoginView.jsx`)
- Clinical single sign-on supporting SAML 2.0 / OIDC institutional providers (Stanford Medicine, Johns Hopkins, Charité).
- Multi-factor hardware attestation selector: FIDO2 / YubiKey, TOTP Authenticator, and mTLS Certificate.

### 14. Access Request Wizard (`RequestAccessView.jsx`)
- Multi-step clinical consortium application form (`SEC-792-REQ`) collecting researcher credentials, institutional affiliations, research protocols, and data protection sign-offs.

---

## 5. Security Dialogs & Interactive Modals

1. **Halt Training Modal (`HaltTrainingModal.jsx`):**
   - Requires explicit typed confirmation (`HALT-CONSORTIUM`) to prevent accidental session interruptions.
   - Automatically revokes hospital gradient ingress tokens and commits a high-priority Merkle audit log.
2. **Byzantine Node Quarantine Modal (`QuarantineModal.jsx`):**
   - Displays offending gradient divergence metric ($L_2$ distance $>3.8\sigma$).
   - Cryptographically excludes the node from the current and upcoming aggregation rounds.
3. **Export Weights Modal (`ExportWeightsModal.jsx`):**
   - Previews model weight archive size, SHA-256 digest, and differential privacy certification ($\epsilon=1.24$).
   - Exports weights in PyTorch (`.pt`), ONNX, or SafeTensors formats.
4. **Switch Role Modal (`SwitchRoleModal.jsx`):**
   - Seamlessly simulates role transitions between `researcher` and `admin` to verify RBAC UI behavior.

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

The application runs at **`http://localhost:5173`** (or next available port).

### Environment Configuration
The application reads its backend API endpoint from `.env`:
```env
VITE_API_URL=http://localhost:5000/api/v1
```
If the backend is not running or is offline, Module 2 automatically operates in high-fidelity mock workstation mode with zero crash risk.
