# Module 11 — Secure Communication (TLS)

> **Status**: In progress. Implemented and verified with automated test suites (5 backend TLS tests,
> 6 hospital-client TLS tests, 3 Flower TLS integration tests; see [Verification](#verification)).
> All network connections that cross a machine boundary are covered. Known limitations and
> architectural boundaries are explicitly documented; items needing further development are
> cataloged under [Future Work](#future-work).

---

## Overview

Module 11 secures every network connection that crosses a machine boundary in the MedFL platform.
Without TLS, bearer tokens (JWTs), model weights, and patient-derived gradients travel in plain text
— a passive eavesdropper on the network can reconstruct the model or replay the token.

Five distinct communication links exist across the system:

| Link | From → To | Protocol | TLS Status | Details |
|------|-----------|----------|------------|---------|
| **[1]** | Browser → Backend (`:5000`) | HTTPS / WSS (`TLSv1.2+`) | ✅ Required | Node.js HTTPS server with explicit `minVersion: 'TLSv1.2'`; Socket.io attached |
| **[2]** | Backend → `api_adapter.py` / Flower server | Subprocess (stdin/stdout) | N/A | Local operating system process spawn; no network sockets |
| **[3]** | Hospital ML service → Backend (`:5000`) | HTTPS REST | ✅ Required | Strict `SSLContext` enforcing `ML_BACKEND_CA` verification; raises `ApiError(502)` with SAN guidance |
| **[4]** | Hospital ML service → Flower server | gRPC over TLS | ✅ Required | `fl.client.start_client` with `root_certificates`; bounded retries; Rule 1: no silent downgrade |
| **[5]** | Hospital operator browser → Hospital ML service | HTTP loopback (`127.0.0.1:8765`) | N/A (Loopback) | W3C Trustworthy Origin; local loopback calls only, CORS-restricted |

---

## Architecture

```
       Admin / Researcher Browser (HTTPS)
                       │
                       │ [1] HTTPS + WSS (TLS 1.2+)
                       ▼
       ┌────────────────────────────────────────────────────────┐
       │ Backend API & Socket.io (Node.js, :5000)                │◄──────────────┐
       └──────────────┬─────────────────────────────────────────┘               │
                      │                                                         │ [3] HTTPS REST
  [2] Subprocess      │ (spawns Flower server                                   │     (CA verified)
      (No network;    │  subprocess per job)                                    │
       No TLS needed) ▼                                                         │
       ┌────────────────────────────────────────────────────────┐               │
       │ Flower Server (federation/flower_app, gRPC)            │               │
       └──────────────────────────▲─────────────────────────────┘               │
                                  │                                             │
                                  │ [4] gRPC over TLS                           │
                                  │     (CA verified)                           │
                                  │                                             │
                      ┌───────────┴─────────────────────────────┐               │
                      │ Hospital Client ML Service (Python)     ├───────────────┘
                      │ (127.0.0.1:8765)                        │
                      └───────────────────▲─────────────────────┘
                                          │
                                          │ [5] HTTP Loopback (127.0.0.1:8765)
                                          │     (W3C Trustworthy Origin; No TLS needed)
                      ┌───────────────────┴─────────────────────┐
                      │ Hospital Operator Browser               │
                      └─────────────────────────────────────────┘
```

---

## Configuration & Environment Variables

To avoid ambiguity, each configuration setting has a single canonical variable name:

| Environment Variable | Where Read | Purpose |
|----------------------|------------|---------|
| `TLS_ENABLED` | `backend/server.js`, `backend/src/services/flowerService.js` | Set to `'true'` to enable HTTPS and Flower TLS across the backend. Defaults to HTTP when unset/false. |
| `TLS_CERT_PATH` | `backend/server.js`, `backend/src/services/flowerService.js` | Filesystem path to the server certificate (PEM format) with SANs covering localhost and server IPs. |
| `TLS_KEY_PATH` | `backend/server.js`, `backend/src/services/flowerService.js` | Filesystem path to the server private key (PEM format). |
| `FLOWER_CA` | `backend/src/services/flowerService.js`, `hospital_client/local_api/server.py` | Filesystem path to the CA certificate used by the Flower server and Flower client for gRPC TLS. |
| `ML_BACKEND_CA` | `hospital_client/local_api/server.py`, `backend/src/services/flowerService.js`, `federation/flower_app/__main__.py` | Filesystem path to the CA certificate used by Python services to verify backend HTTPS requests. |
| `ML_BACKEND_URL` | `hospital_client/local_api/server.py`, `federation/flower_app/__main__.py` | Base URL of the backend API (e.g. `https://localhost:5000/api/v1`). |

---

## Components

### 1. Certificate Tooling — `scripts/make_dev_certs.py`

Generates a 2048-bit RSA local CA and a signed server certificate with Subject Alternative Names (SANs):

```bash
# Generate dev certs into gitignored certs/ directory
python scripts/make_dev_certs.py --out-dir certs

# With optional LAN IP or extra host for multi-machine setups
python scripts/make_dev_certs.py --out-dir certs --lan-ip 192.168.1.100
```

**Security Invariants**:
- Overwrite protection: Refuses to overwrite existing certificates without `--force`.
- Private key protection: Private keys are never printed to stdout; restrictive permissions (`0o600`) are applied on disk.
- SAN enforcement: Always includes `localhost`, `127.0.0.1`, and `::1`.

**Windows Browser Trust**:
```powershell
# Trust the local dev CA (run once in an elevated or user shell)
certutil -user -addstore Root certs\ca.pem

# Remove/untrust the local dev CA
certutil -user -delstore Root "MedFL Local Development CA"
```

### 2. Backend HTTPS Server — `backend/server.js`

When `TLS_ENABLED=true`, `createHttpServer()` starts an HTTPS server using `https.createServer`:
- Requires `TLS_CERT_PATH` and `TLS_KEY_PATH`.
- Strict validation: If either file is missing or unreadable, the server immediately throws `[TLS ERROR]` and refuses to start (no silent fallback).
- Configures explicit `minVersion: 'TLSv1.2'`.
- Socket.io attaches to the HTTPS server and automatically operates over `wss://`.

### 3. Flower Server Subprocess Spawning — `backend/src/services/flowerService.js`

When `TLS_ENABLED=true`:
- Requires `FLOWER_CA`, `TLS_CERT_PATH`, and `TLS_KEY_PATH`. If any are unset or the files do not exist, `flowerService.start()` immediately rejects with `[TLS ERROR]` (no plaintext Flower server in TLS mode).
- Resolves these paths to absolute paths, because the Flower process runs from the repository root, then passes `--ca-cert`, `--server-cert`, and `--server-key` to `python -m federation.flower_app`.
- Passes `--backend-ca` (pointing to `ML_BACKEND_CA`) for backend identity verification over HTTPS.

### 4. Flower Server CLI — `federation/flower_app/__main__.py`

- Requires all three of `--ca-cert`, `--server-cert`, and `--server-key` if any are passed. Specifying a partial set causes immediate termination with exit code 1.
- Strict `backend_identity_check`: If `--backend-ca` is provided, verifies that the file exists; if missing, raises `FileNotFoundError` rather than silently falling back to system CAs.

### 5. Hospital ML Service — `hospital_client/local_api/server.py`

- **Backend TLS verification**: All HTTP/HTTPS requests go through `backend_urlopen()`. For HTTPS URLs, a strict `SSLContext` enforcing `ML_BACKEND_CA` is used (`verify_mode = ssl.CERT_REQUIRED`, `check_hostname = True`). Verification failures raise `ApiError(502)` with SAN guidance.
- **Rule 1 (No silent downgrade)**: In `r_flower_connect`, if `BACKEND_URL` starts with `https://`, Flower TLS is mandatory.
- **Local config only**: The CA certificate path is resolved strictly from local configuration (`FLOWER_CA` / `ML_BACKEND_CA` / `certs/ca.pem`). Request bodies are **not** permitted to supply a `ca_path` (`body.get("ca_path")` is removed and ignored), preventing an attacker who controls the server address from also dictating the trusted CA.
- **Startup warning**: If `ML_BACKEND_URL` is plain HTTP and points to a non-loopback host, logs a prominent startup security warning.

---

## Known Limitations

- **Backend CORS and Socket.io accept any origin**: The backend and Socket.io currently accept requests from any origin (`cors()`). Restricting origins to specific frontend origins is left for production deployment.
- **Real browser verification is partly done**: The CA install needs a Windows security dialog, so this check is manual. Confirmed by hand: the HTTPS API loads without a certificate warning, and Socket.io connects over WSS. Still open: the certificate viewer check, confirming `http://` is refused, the hospital pages on `127.0.0.1:8765`, and a complete federation run over TLS.
- **No mutual TLS (mTLS)**: The server verifies itself to clients, but clients do not present client certificates to the server (authentication is achieved via bearer JWTs over TLS).
- **No certificate rotation without restart**: Updating certificates requires restarting the backend and Flower server.
- **Browser → Local ML Service is HTTP loopback**: Served on `http://127.0.0.1:8765`. Modern browsers permit HTTPS pages to fetch from `127.0.0.1` because loopback is a W3C Trustworthy Origin.

---

## Future Work

- **mTLS**: Hospital-specific client certificates verified by Flower server and backend.
- **Automated ACME / Let's Encrypt**: Automated issuance and renewal for public server domains.
- **OCSP Stapling / CRL Checking**: Strict revocation verification.
- **HSTS Header**: `Strict-Transport-Security` enforcement on the backend API.

---

## Verification

All tests run without requiring pre-existing `certs/` on disk (certificates are generated dynamically into temporary directories inside test fixtures):

### 1. Backend Tests (`cd backend && npm test`)
- **188 passed, 5 test suites passed** (183 existing + 5 in `tests/api/tls.test.js`):
  - `creates standard HTTP server by default when TLS is disabled`
  - `refuses to start when TLS is enabled but cert or key path is missing`
  - `refuses to start when TLS certificate file does not exist`
  - `creates HTTPS server with explicit TLSv1.2 when valid cert and key are provided (never skips)`
  - `refuses to start Flower server in TLS mode when Flower TLS certs are missing or incomplete`

### 2. Hospital Client Tests (`python -m pytest hospital_client/local_api/tests/test_tls.py -v`)
- **6 passed, 0 skipped**:
  - `test_security_warning_on_remote_http`
  - `test_ssl_context_enforces_ca`
  - `test_backend_urlopen_with_trusted_ca`
  - `test_backend_urlopen_with_wrong_ca_fails`
  - `test_rule_1_flower_no_silent_downgrade`
  - `test_flower_connect_rejects_body_ca_path`

### 3. Federation Tests (`python -m pytest federation/tests/test_flower_tls.py -v`)
- **3 passed, 0 skipped**:
  - `test_flower_tls_round_success` — full round over TLS in main thread.
  - `test_flower_tls_wrong_ca_fails_bounded` — Flower server in subprocess with TLS; client with untrusted CA fails with SSL verification error within bound.
  - `test_flower_tls_insecure_client_fails_bounded` — Flower server in subprocess with TLS; client with insecure=True fails within bound.
