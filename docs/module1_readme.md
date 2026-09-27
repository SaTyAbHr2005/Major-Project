# Module 1: Authentication and Role Management

**Project:** Secure and Privacy-Preserving Federated Deep Learning Training Platform for Medical Imaging  
**Owner:** Member 3 (Backend + Database + Security Infrastructure)  
**Branch:** `feature/module-01-auth`  
**Last Updated:** 2026-09-07

---

## Status

- [x] User schema complete
- [x] Hospital schema complete
- [x] Role definitions complete
- [x] Permission definitions complete
- [x] Registration implemented
- [x] Login implemented
- [x] Logout implemented
- [x] JWT authentication implemented
- [x] RBAC implemented
- [x] Hospital access control implemented
- [x] Account approval workflow implemented
- [x] Audit logging implemented
- [x] API validation implemented
- [x] Error handling implemented
- [x] Database indexes implemented
- [x] Unit tests completed
- [x] Integration tests completed
- [x] Authentication/security-focused automated tests completed
- [x] Documentation completed
- [x] **Module 1 functionally complete**

> **Note:** The Module 1 automated security/authentication test suite currently passes 29/29 tests. Additional manual penetration testing and input-fuzzing are recommended as a project-level security hardening activity, but they are not required to mark the implemented Module 1 functionality complete.

---

## 1. Purpose

Module 1 is the authentication and authorization foundation for the entire platform.

Every other module (FL Engine, Desktop App, Hospital App, Mobile App, Monitoring) depends on Module 1 to verify who a user is and what they are allowed to do. No module is permitted to implement its own user session, role check, or hospital isolation logic — they must consume the interfaces provided here.

Module 1 is responsible for:
- Establishing identity (who are you?)
- Establishing authorization (what are you allowed to do?)
- Enforcing hospital isolation (which hospital can you access?)
- Recording all authentication and security events in an immutable audit trail

---

## 2. Responsibilities

| Responsibility | Handled by Module 1 |
|---|---|
| User registration | ✅ |
| Secure password storage (bcrypt) | ✅ |
| JWT issuance and validation | ✅ |
| Role-based access control (RBAC) | ✅ |
| Hospital operator isolation | ✅ |
| Admin account approval workflow | ✅ |
| Audit log creation | ✅ |
| Failed login tracking and temporary lockout | ✅ |
| Preventing plaintext passwords in storage or logs | ✅ |
| Preventing admin creation via public endpoint | ✅ |

**Not in Module 1 scope:**
- Dataset access control (Module 4/5)
- Training session authorization (Module 9)
- Model access (Module 6/15)
- Monitoring dashboard (Module 13)
- Researcher-hospital participation rules (future marketplace)

---

## 3. Roles

### Role Definitions

```
ROLES.ADMIN            = 'admin'
ROLES.RESEARCHER       = 'researcher'
ROLES.HOSPITAL_OPERATOR = 'hospital_operator'
```

Defined in: `src/authentication/permissions/roles.js`

### Permission Table

| Permission | admin | researcher | hospital_operator |
|---|:---:|:---:|:---:|
| `manage_users` | ✅ | ❌ | ❌ |
| `approve_researchers` | ✅ | ❌ | ❌ |
| `manage_hospitals` | ✅ | ❌ | ❌ |
| `manage_hospital_users` | ✅ | ❌ | ❌ |
| `view_audit_logs` | ✅ | ❌ | ❌ |
| `view_security_events` | ✅ | ❌ | ❌ |
| `login` | ✅ | ✅ | ✅ |
| `view_authorized_hospitals` | ❌ | ✅ | ❌ |
| `manage_local_client` | ❌ | ❌ | ✅ |
| `view_hospital_status` | ❌ | ❌ | ✅ |

> **Important:** Researcher access to specific hospitals, models, and training sessions is NOT automatically granted. Those resource-specific controls will be implemented in the relevant future modules (Module 9, 6, etc.).

### Account Status and Transitions

| Status | Meaning |
|---|---|
| `pending` | Registered but awaiting admin approval |
| `active` | Approved — can log in and use the platform |
| `suspended` | Temporarily blocked by an admin |
| `rejected` | Registration was rejected by admin |
| `deactivated` | Administratively deactivated |

**Transition rules:**

```
Registration → pending
Admin approve → active
Admin reject  → rejected
Admin suspend → suspended
Admin deactivate → deactivated
```

Only users with status `active` may authenticate successfully.

**Admin creation:** Admin accounts are NOT created via the public registration endpoint. They must be provisioned through a controlled administrative mechanism (e.g., a seed script or secure CLI command). Attempting to register an `admin` role via `POST /api/v1/auth/register` returns `403 FORBIDDEN`.

---

## 4. Identifiers

These identifiers are frozen project-wide contracts. Do NOT rename them.

| Identifier | Type | Description |
|---|---|---|
| `user_id` | String | Unique identifier for every user. Format: `USR_<12-hex-chars>`. e.g. `USR_a3b4c5d6e7f8` |
| `hospital_id` | String | Unique identifier for a hospital record. e.g. `HOSP_000001` |
| `client_id` | String | Identifier for a Flower FL client connected to a hospital. Stored in `hospital.client_ids[]` |

### Relationships

```
User (hospital_operator)
  └── hospital_id ──► Hospital
                         └── client_ids[] ──► FL Client (Module 9)
```

- A `hospital_operator` has exactly one `hospital_id`.
- A `Hospital` record may have multiple `client_ids` (design is extensible — not hardcoded to one client).
- `researcher` and `admin` users have `hospital_id = null`.
- The `client_id` is managed by Module 9 (Federated Engine). Module 1 only stores them as strings on the Hospital document.

---

## 5. Database Schemas

### `users` Collection

File: `src/authentication/models/User.js`

| Field | Type | Required | Indexed | Notes |
|---|---|:---:|:---:|---|
| `user_id` | String | ✅ | ✅ unique | Project-wide frozen identifier |
| `name` | String | ✅ | — | Display name |
| `email` | String | ✅ | ✅ unique | Login credential — must be unique |
| `password_hash` | String | ✅ | — | bcrypt hash. `select: false` — never returned by queries by default |
| `role` | String enum | ✅ | — | `researcher` \| `hospital_operator` \| `admin` |
| `hospital_id` | String | conditional | — | Required for `hospital_operator`. `null` for others |
| `status` | String enum | ✅ | — | `pending` \| `active` \| `suspended` \| `rejected` \| `deactivated` |
| `last_login_at` | Date | — | — | Updated on every successful login |
| `failed_login_attempts` | Number | — | — | Incremented on wrong password. Reset on success |
| `lock_until` | Date | — | — | Temporary lockout timestamp. Set when attempts exceed `MAX_FAILED_LOGIN_ATTEMPTS` |
| `created_at` | Date | — | — | Auto-managed by Mongoose timestamps |
| `updated_at` | Date | — | — | Auto-managed by Mongoose timestamps |

**Security notes:**
- `password_hash` has `select: false` — it is **never** included in query results unless explicitly selected with `.select('+password_hash')`.
- The `pre('save')` hook automatically hashes `password_hash` using bcrypt (salt rounds: 10) when the field is modified.
- `password_hash` is NEVER returned by any API response.

### `hospitals` Collection

File: `src/authentication/models/Hospital.js`

| Field | Type | Required | Indexed | Notes |
|---|---|:---:|:---:|---|
| `hospital_id` | String | ✅ | ✅ unique | Project-wide frozen identifier |
| `name` | String | ✅ | — | Hospital display name |
| `status` | String enum | ✅ | — | `pending` \| `active` \| `suspended` \| `inactive` |
| `client_ids` | [String] | — | — | Array of FL client identifiers. One per hospital initially, but extensible |
| `created_at` | Date | — | — | Auto-managed |
| `updated_at` | Date | — | — | Auto-managed |

> **Rule:** This collection must NEVER store raw medical images, datasets, or patient records.

### `audit_logs` Collection

File: `src/authentication/models/AuditLog.js`

| Field | Type | Required | Indexed | Notes |
|---|---|:---:|:---:|---|
| `audit_id` | String | ✅ | ✅ unique | Format: `AUD_<24-hex-chars>` |
| `user_id` | String | — | ✅ | Who triggered the event. `UNKNOWN` for anonymous attempts |
| `action` | String | ✅ | — | See audit events table below |
| `resource_type` | String | — | — | e.g. `User`, `Hospital` |
| `resource_id` | String | — | — | e.g. the affected `user_id` |
| `hospital_id` | String | — | ✅ | Hospital context for the event |
| `ip_address` | String | — | — | Requester IP address |
| `user_agent` | String | — | — | Requester browser/client string |
| `timestamp` | Date | ✅ | ✅ | Event time |
| `success` | Boolean | ✅ | — | Whether the action succeeded |
| `metadata` | Object | — | — | Additional context. Sensitive keys are stripped before storage |

**Immutability rule:** Audit logs are **append-only**. Module 1 exposes no endpoint to update or delete audit records.

---

## 6. API Contract

**Base URL:** `/api/v1`

**Standard success response:**
```json
{
  "success": true,
  "data": { ... }
}
```

**Standard error response:**
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable description"
  }
}
```

### Error Codes Reference

| Code | HTTP | Meaning |
|---|---|---|
| `INVALID_CREDENTIALS` | 401 | Wrong password or unknown email |
| `ACCOUNT_PENDING` | 403 | Account not yet approved |
| `ACCOUNT_SUSPENDED` | 403 | Account suspended by admin |
| `ACCOUNT_REJECTED` | 403 | Account registration was rejected |
| `ACCOUNT_DEACTIVATED` | 403 | Account has been deactivated |
| `UNAUTHORIZED` | 401 | No token provided |
| `INVALID_TOKEN` | 401 | Token is malformed or invalid |
| `TOKEN_EXPIRED` | 401 | Token has expired |
| `FORBIDDEN` | 403 | Role does not have permission |
| `USER_NOT_FOUND` | 401 | Token valid but user no longer exists |
| `EMAIL_ALREADY_EXISTS` | 400 | Duplicate email on registration |
| `INVALID_ROLE` | 400 | Unrecognized role value |
| `HOSPITAL_ACCESS_DENIED` | 403 | Operator trying to access another hospital |
| `VALIDATION_ERROR` | 400 | Missing or invalid request fields |
| `SERVER_ERROR` | 500 | Unexpected server error |

---

### `POST /api/v1/auth/register`

Register a new `researcher` or `hospital_operator` account.

- **Auth required:** No
- **Allowed roles:** Public
- **Admin cannot be created here** — returns `403 FORBIDDEN`

**Request body:**
```json
{
  "name": "Dr. Jane Smith",
  "email": "jane@hospital.org",
  "password": "securepassword",
  "role": "hospital_operator",
  "hospital_id": "HOSP_000001"
}
```

> `hospital_id` is required when `role = hospital_operator` and must reference an existing Hospital document.  
> `hospital_id` must be omitted (or absent) when `role = researcher`.

**Success response (201):**
```json
{
  "success": true,
  "data": {
    "user": {
      "user_id": "USR_a3b4c5d6e7f8",
      "name": "Dr. Jane Smith",
      "email": "jane@hospital.org",
      "role": "hospital_operator",
      "hospital_id": "HOSP_000001",
      "status": "pending"
    }
  }
}
```

**Possible errors:** `VALIDATION_ERROR`, `FORBIDDEN` (admin role), `INVALID_ROLE`, `EMAIL_ALREADY_EXISTS`

---

### `POST /api/v1/auth/login`

Authenticate a user and receive a JWT access token.

- **Auth required:** No
- **Allowed roles:** All roles (if status is `active`)

**Request body:**
```json
{
  "email": "jane@hospital.org",
  "password": "securepassword"
}
```

**Success response (200):**
```json
{
  "success": true,
  "data": {
    "user": {
      "user_id": "USR_a3b4c5d6e7f8",
      "name": "Dr. Jane Smith",
      "email": "jane@hospital.org",
      "role": "hospital_operator",
      "hospital_id": "HOSP_000001",
      "status": "active"
    },
    "access_token": "<JWT>"
  }
}
```

> `password_hash` is NEVER included in this response.

**Possible errors:** `VALIDATION_ERROR`, `INVALID_CREDENTIALS`, `ACCOUNT_PENDING`, `ACCOUNT_SUSPENDED`, `ACCOUNT_REJECTED`, `ACCOUNT_DEACTIVATED`

---

### `POST /api/v1/auth/logout`

Log out the authenticated user. Records a `LOGOUT` audit event.

- **Auth required:** Yes (`requireAuth`)
- **Allowed roles:** All active roles

> Logout is **stateless**. The server does not maintain a token blacklist. The client is responsible for discarding the token. The JWT will remain technically valid until its `exp` claim. See Section 7 for details.

**Success response (200):**
```json
{
  "success": true,
  "data": {
    "message": "Logged out successfully"
  }
}
```

---

### `GET /api/v1/auth/me`

Retrieve the authenticated user's profile.

- **Auth required:** Yes (`requireAuth`)
- **Allowed roles:** All active roles

**Success response (200):**
```json
{
  "success": true,
  "data": {
    "user": {
      "user_id": "USR_a3b4c5d6e7f8",
      "name": "Dr. Jane Smith",
      "email": "jane@hospital.org",
      "role": "hospital_operator",
      "hospital_id": "HOSP_000001",
      "status": "active"
    }
  }
}
```

**Possible errors:** `UNAUTHORIZED`, `INVALID_TOKEN`, `TOKEN_EXPIRED`

---

### `POST /api/v1/auth/change-password`

Change the authenticated user's own password.

- **Auth required:** Yes (`requireAuth`)
- **Allowed roles:** All active roles

**Request body:**
```json
{
  "currentPassword": "oldpassword",
  "newPassword": "newstrongpassword"
}
```

**Success response (200):**
```json
{
  "success": true,
  "data": {
    "message": "Password changed successfully"
  }
}
```

**Possible errors:** `VALIDATION_ERROR` (missing/empty password fields or invalid password change request), `INVALID_CREDENTIALS` (wrong current password), `UNAUTHORIZED`

---

### `POST /api/v1/admin/users/:user_id/approve`

Approve a pending user account (sets status to `active`).

- **Auth required:** Yes (`requireAuth`)
- **Allowed roles:** `admin` only

**Success response (200):**
```json
{
  "success": true,
  "data": {
    "message": "User status updated to active",
    "user": {
      "user_id": "USR_a3b4c5d6e7f8",
      "status": "active"
    }
  }
}
```

**Possible errors:** `UNAUTHORIZED`, `FORBIDDEN`, `USER_NOT_FOUND`

---

### `POST /api/v1/admin/users/:user_id/reject`

Reject a pending user account (sets status to `rejected`).

- **Auth required:** Yes (`requireAuth`)
- **Allowed roles:** `admin` only

**Success/error format:** Same structure as approve.

---

### `POST /api/v1/admin/users/:user_id/suspend`

Suspend an active user account (sets status to `suspended`).

- **Auth required:** Yes (`requireAuth`)
- **Allowed roles:** `admin` only

**Success/error format:** Same structure as approve.

---

## 7. JWT

### Token Payload

```json
{
  "sub": "USR_a3b4c5d6e7f8",
  "user_id": "USR_a3b4c5d6e7f8",
  "role": "hospital_operator",
  "hospital_id": "HOSP_000001",
  "iat": 1725700000,
  "exp": 1725703600
}
```

| Claim | Purpose |
|---|---|
| `sub` | Standard JWT subject — same as `user_id` |
| `user_id` | Frozen project identifier — used by all modules |
| `role` | Role string used for middleware authorization |
| `hospital_id` | Used by `requireHospitalAccess` for isolation checks |
| `iat` | Issued-at timestamp (set automatically by `jsonwebtoken`) |
| `exp` | Expiry timestamp (set by `JWT_EXPIRES_IN`) |

> Personal information (name, email) is NOT stored in the JWT payload.  
> Medical information is NEVER stored in the JWT payload.

### Expiration

- Default: `1h` (one hour)
- Controlled by: `JWT_EXPIRES_IN` environment variable
- Short expiry is intentional — it reduces the exposure window if a token is intercepted.

### Secret Management

- The signing secret is loaded exclusively from `process.env.JWT_SECRET`.
- If `JWT_SECRET` is not set, `generateToken()` throws an error immediately — it never falls back to a default value.
- Never hardcode the secret. Never commit `.env` to git.

### Logout / Invalidation Strategy

JWT logout in Module 1 is **stateless**:

1. The client discards the access token from local storage / memory.
2. The server records a `LOGOUT` audit event.
3. No token blacklist or Redis revocation store is used.

**Known limitation:** A stateless JWT cannot be forcibly invalidated before its `exp` timestamp. If a token is stolen after logout, it remains technically valid until expiry.

**Future enhancement (out of scope for Module 1):** A refresh token system with server-side revocation can be layered on top. This requires a persistent token store (e.g., Redis). This is NOT implemented in Module 1 and must not be added without a team-wide decision.

### Token Validation

`requireAuth` middleware performs:
1. Extracts `Bearer <token>` from the `Authorization` header.
2. Calls `jwt.verify(token, process.env.JWT_SECRET)`.
3. On `TokenExpiredError`, returns `TOKEN_EXPIRED`.
4. On any other JWT error, returns `INVALID_TOKEN`.
5. Re-fetches the user from MongoDB using `decoded.user_id` to confirm the user still exists and is `active`.
6. Attaches the user document to `req.user`.

---

## 8. RBAC

### How It Works

Role checking is centralized in two files — **never scattered through business logic**:

- `src/authentication/permissions/roles.js` — defines role strings and permission arrays.
- `src/authentication/middleware/roleMiddleware.js` — implements `requireRole()`.

### `requireRole(...roles)`

```js
// Usage — only admin can call this route:
router.post('/users/:user_id/approve', requireAuth, requireRole('admin'), approveUser);

// Usage — admin or researcher:
router.get('/models', requireAuth, requireRole('admin', 'researcher'), listModels);
```

- Reads `req.user.role` (set by `requireAuth`).
- Returns `403 FORBIDDEN` with code `FORBIDDEN` if the role is not in the allowed list.
- Must always be used **after** `requireAuth`.

### `requireAuth`

Always apply `requireAuth` first. It validates the JWT, checks account status, and populates `req.user`. Without it, `requireRole` and `requireHospitalAccess` will not have `req.user` to work with.

```js
// Always in this order:
router.use(requireAuth, requireRole('admin'));
```

---

## 9. Hospital Isolation

This is a critical security requirement.

### Rule

A `hospital_operator` with `hospital_id = HOSP_000001` must NEVER be able to access resources belonging to `HOSP_000002` or any other hospital.

### Implementation

File: `src/authentication/middleware/hospitalMiddleware.js`

```js
// Example protected route in a future module:
router.get('/hospitals/:hospital_id/clients', requireAuth, requireHospitalAccess, getClients);
```

### `requireHospitalAccess` Logic

1. Reads `hospital_id` from `req.params.hospital_id` or `req.body.hospital_id`.
2. Returns `VALIDATION_ERROR` if `hospital_id` is not present in the request.
3. If `req.user.role === 'admin'` → **passes through** (admin can access all hospitals).
4. If `req.user.role === 'hospital_operator'` → **compares** `req.user.hospital_id` against the requested `hospital_id`:
   - Match → passes through.
   - No match → returns `403 HOSPITAL_ACCESS_DENIED`.
5. If `req.user.role === 'researcher'` → **blocked** at Module 1 level. Resource-specific researcher participation rules are implemented in future modules (Module 9, etc.).

### Preferred URL Pattern for Hospital Resources

Always put `hospital_id` in the route parameter for hospital-specific resources:

```
/api/v1/hospitals/:hospital_id/clients
/api/v1/hospitals/:hospital_id/status
/api/v1/hospitals/:hospital_id/training
```

This allows `requireHospitalAccess` to resolve the requested hospital automatically.

---

## 10. Audit Logging

All authentication and authorization events are logged using `AuditService.logEvent()`.

### Audit Events

| Action | Trigger |
|---|---|
| `REGISTER_SUCCESS` | Successful new user registration |
| `REGISTER_FAILED` | (Future) Failed registration attempt |
| `LOGIN_SUCCESS` | Successful login |
| `LOGIN_FAILED` | Wrong password, unknown email, locked account, or non-active account |
| `LOGOUT` | Successful logout request |
| `PASSWORD_CHANGED` | User successfully changes their password |
| `USER_APPROVED` | Admin approves a pending account |
| `USER_REJECTED` | Admin rejects a pending account |
| `USER_SUSPENDED` | Admin suspends an active account |
| `ACCESS_DENIED` | (Available for future modules to log) |

### Security Rules for Audit Metadata

The following fields are **never stored** in `metadata`:
- Passwords or password hashes
- JWT tokens
- JWT secrets
- Encryption keys

`AuditService.logEvent()` automatically strips keys named `password`, `password_hash`, `token`, `jwt`, and `secret` from any metadata object before saving.

### Usage by Future Modules

```js
const AuditService = require('../authentication/services/auditService');

await AuditService.logEvent({
  user_id: req.user.user_id,
  action: 'ACCESS_DENIED',
  resource_type: 'TrainingSession',
  resource_id: session_id,
  hospital_id: req.user.hospital_id,
  ip_address: req.ip,
  user_agent: req.get('User-Agent'),
  success: false,
  metadata: { reason: 'unauthorized_hospital' }
});
```

---

## 11. Security

### Password Hashing

- Library: `bcrypt` (npm package `bcrypt`)
- Salt rounds: `10`
- Hashing is performed automatically in the Mongoose `pre('save')` hook on the `User` model.
- Passwords are **never** stored in plaintext.
- Passwords are **never** logged.
- `password_hash` has `select: false` — excluded from all queries by default.

### Failed Login and Lockout

- `failed_login_attempts` is incremented after every wrong-password attempt.
- When `failed_login_attempts >= MAX_FAILED_LOGIN_ATTEMPTS` (env var, default: 5), `lock_until` is set to `now + 15 minutes`.
- Locked accounts receive `INVALID_CREDENTIALS` — no indication given to the requester as to whether the account is locked or the password was wrong (prevents enumeration).
- On successful login, `failed_login_attempts` is reset to `0` and `lock_until` is cleared.

### Token Security

- JWT is signed with `HS256` using `process.env.JWT_SECRET`.
- Secret is validated at token generation time — missing secret throws immediately.
- Never hardcode JWT secrets. Use `.env` locally and secrets management in production.

### Validation Rules

| Field | Rule |
|---|---|
| `email` | Must be a string. Must be unique. |
| `password` | Must be provided. Never logged. |
| `role` | Must be one of `researcher`, `hospital_operator`, `admin`. Admin blocked at registration. |
| `hospital_id` | Required for `hospital_operator`. Must reference an existing Hospital document. Must be absent for `researcher`. |
| `status` | Controlled by server only. Never set by client. |

### Sensitive Data Restrictions

- No medical images, patient records, or medical datasets in any Module 1 collection.
- No ML model weights stored here.
- MongoDB documents for `users`, `hospitals`, `audit_logs` are authentication/identity data only.
- Frontend never accesses MongoDB directly — only via the REST API.

### Secrets

The following must be kept in environment variables and never committed to Git:

```
JWT_SECRET
MONGODB_URI
```

---

## 12. File Structure

```
backend/
├── server.js                          # Entry point — starts Express + MongoDB
├── .env.example                       # Environment variable template (commit this)
├── .gitignore                         # Excludes .env and node_modules
├── package.json                       # Dependencies and Jest config
│
└── src/
    ├── app.js                         # Express app — routes wired
    │
    ├── database/
    │   └── connection/
    │       └── db.js                  # Mongoose connection to MongoDB
    │
    └── authentication/
        ├── models/
        │   ├── User.js               # users collection schema
        │   ├── Hospital.js           # hospitals collection schema
        │   └── AuditLog.js          # audit_logs collection schema
        │
        ├── controllers/
        │   ├── authController.js     # register, login, logout, me, changePassword
        │   └── adminController.js    # approve, reject, suspend
        │
        ├── routes/
        │   ├── authRoutes.js         # /api/v1/auth/*
        │   └── adminRoutes.js        # /api/v1/admin/*
        │
        ├── middleware/
        │   ├── authMiddleware.js     # requireAuth()
        │   ├── roleMiddleware.js     # requireRole()
        │   └── hospitalMiddleware.js # requireHospitalAccess()
        │
        ├── services/
        │   ├── authService.js        # generateToken(), generateUserId()
        │   └── auditService.js       # AuditService.logEvent()
        │
        ├── validators/
        │   └── authValidator.js      # Placeholder for advanced schema validation
        │
        └── permissions/
            └── roles.js              # ROLES, ROLE_PERMISSIONS constants

tests/
└── api/
    └── auth.test.js                  # Full integration test suite (Jest + Supertest)
```

---

## 13. Testing

**Test runner:** Jest + Supertest  
**In-memory DB:** mongodb-memory-server (no external MongoDB required for tests)

**Run command:**
```bash
npm test
```

### Verified Test Results

The complete Module 1 authentication and authorization test suite has been executed successfully.

```text
Test Suites: 1 passed, 1 total
Tests:       29 passed, 29 total
Snapshots:   0 total
Time:        7.959 s
```

**Result: 29/29 tests passed ✅**

### Test Coverage

| # | Test Case | Group | Status |
|---|---|---|:---:|
| 1 | Successfully register a researcher | Registration | ✅ Passed |
| 2 | Reject admin registration via public endpoint | Registration | ✅ Passed |
| 3 | Reject admin registration containing hospital_id | Registration | ✅ Passed |
| 4 | Reject hospital_operator without hospital_id | Registration | ✅ Passed |
| 5 | Reject researcher with unexpected hospital_id | Registration | ✅ Passed |
| 6 | Reject duplicate email registration | Registration | ✅ Passed |
| 7 | Allow active user to login | Login / Status | ✅ Passed |
| 8 | Reject wrong password and increment attempts | Login / Status | ✅ Passed |
| 9 | Temporarily lock account after maximum failed login attempts | Login / Status | ✅ Passed |
| 10 | Deny login for pending / suspended / rejected / deactivated accounts | Login / Status | ✅ Passed |
| 11 | Deny login for nonexistent email | Login / Status | ✅ Passed |
| 12 | Accept valid JWT token | JWT | ✅ Passed |
| 13 | Reject missing token | JWT | ✅ Passed |
| 14 | Reject malformed token | JWT | ✅ Passed |
| 15 | Reject expired JWT token | JWT | ✅ Passed |
| 16 | Reject JWT signed with the wrong secret | JWT | ✅ Passed |
| 17 | Reject modified admin role with invalid signature | JWT Security | ✅ Passed |
| 18 | Successfully log out and create audit event | Logout | ✅ Passed |
| 19 | Successfully change password | Change Password | ✅ Passed |
| 20 | Reject change-password with incorrect current password | Change Password | ✅ Passed |
| 21 | Reject change-password without authentication | Change Password | ✅ Passed |
| 22 | Reject empty new password with validation error | Change Password / Validation | ✅ Passed |
| 23 | Allow admin to approve pending user | RBAC | ✅ Passed |
| 24 | Reject researcher from accessing admin endpoint | RBAC | ✅ Passed |
| 25 | Operator can access their own hospital | Hospital Isolation | ✅ Passed |
| 26 | Operator cannot access another hospital | Hospital Isolation | ✅ Passed |
| 27 | Second operator can access their own hospital | Hospital Isolation | ✅ Passed |
| 28 | Strip sensitive keys from audit metadata | Audit Log Security | ✅ Passed |
| 29 | Do not store password hash in normal API responses | Data Exposure Security | ✅ Passed |

### Security Test Coverage

The automated suite verifies the following security-sensitive behaviors:

- JWT signature validation
- Expired-token rejection
- Malformed-token rejection
- Wrong-secret rejection
- JWT role-tampering rejection
- Authentication bypass prevention
- Role-based authorization
- Hospital-level isolation
- Failed-login tracking
- Temporary account lockout
- Password-change validation
- Password hash non-disclosure
- Sensitive audit metadata stripping
- Nonexistent-user login handling without account enumeration

### Manual Security Hardening

The automated tests above are complete for the current Module 1 implementation. Additional manual penetration testing, input-fuzzing, deployment configuration review, and production security hardening can be performed later as part of the project's overall security validation.

## 14. Integration Contracts

This section is for **other module developers**. These are the exact interfaces you can import and use.

### `requireAuth`

```js
const { requireAuth } = require('../authentication/middleware/authMiddleware');
```

- **Apply as:** Route middleware
- **Reads:** `Authorization: Bearer <token>` header
- **Populates:** `req.user` (full Mongoose User document, minus `password_hash`)
- **Rejects with:** `401 UNAUTHORIZED`, `401 INVALID_TOKEN`, `401 TOKEN_EXPIRED`, `401 USER_NOT_FOUND`, `403 ACCOUNT_*`
- **Must be applied before** `requireRole` or `requireHospitalAccess`

```js
// Example
router.get('/training', requireAuth, getTrainingStatus);
```

### `requireRole(...roles)`

```js
const { requireRole } = require('../authentication/middleware/roleMiddleware');
```

- **Apply as:** Route middleware (after `requireAuth`)
- **Reads:** `req.user.role`
- **Arguments:** One or more role strings from `ROLES`
- **Rejects with:** `403 FORBIDDEN`

```js
// Example — admin only
router.post('/sessions', requireAuth, requireRole('admin'), createSession);

// Example — admin or researcher
router.get('/models', requireAuth, requireRole('admin', 'researcher'), listModels);
```

### `requireHospitalAccess`

```js
const { requireHospitalAccess } = require('../authentication/middleware/hospitalMiddleware');
```

- **Apply as:** Route middleware (after `requireAuth`)
- **Reads:** `req.params.hospital_id` or `req.body.hospital_id`
- **Behavior:**
  - `admin` → always passes
  - `hospital_operator` → passes only if `req.user.hospital_id === req.params.hospital_id`
  - `researcher` → blocked (resource-specific rules must be added by the relevant future module)
- **Rejects with:** `403 HOSPITAL_ACCESS_DENIED`, `400 VALIDATION_ERROR`

```js
// Example — hospital-specific resource (preferred URL pattern)
router.get('/hospitals/:hospital_id/clients', requireAuth, requireHospitalAccess, getClients);
```

### `ROLES` constants

```js
const { ROLES } = require('../authentication/permissions/roles');

// ROLES.ADMIN            → 'admin'
// ROLES.RESEARCHER       → 'researcher'
// ROLES.HOSPITAL_OPERATOR → 'hospital_operator'
```

Always use `ROLES.*` constants instead of hardcoding role strings in future module code.

### `AuditService.logEvent(params)`

```js
const AuditService = require('../authentication/services/auditService');

await AuditService.logEvent({
  user_id: req.user.user_id,       // string — required
  action: 'ACCESS_DENIED',         // string — required
  resource_type: 'TrainingSession', // string — optional
  resource_id: session_id,          // string — optional
  hospital_id: req.user.hospital_id, // string — optional
  ip_address: req.ip,               // string — optional
  user_agent: req.get('User-Agent'), // string — optional
  success: false,                   // boolean — required
  metadata: { reason: '...' }       // object — optional, sensitive keys auto-stripped
});
```

---

## 15. Dependencies

### Runtime

| Package | Version | Purpose |
|---|---|---|
| `express` | ^5.2.1 | HTTP server framework |
| `mongoose` | ^9.9.5 | MongoDB ODM |
| `bcrypt` | ^6.0.0 | Password hashing |
| `jsonwebtoken` | ^9.0.3 | JWT generation and verification |
| `dotenv` | ^17.4.2 | Environment variable loading |
| `cors` | ^2.8.6 | Cross-origin resource sharing |

### Development

| Package | Version | Purpose |
|---|---|---|
| `jest` | ^30.5.1 | Test runner |
| `supertest` | ^7.2.2 | HTTP integration testing |
| `cross-env` | ^10.1.0 | Cross-platform environment variable injection |
| `mongodb-memory-server` | ^11.x | In-memory MongoDB for tests (no external DB needed) |

### Environment Variables

| Variable | Required | Default | Description |
|---|:---:|---|---|
| `MONGODB_URI` | ✅ | — | MongoDB connection string |
| `JWT_SECRET` | ✅ | — | JWT signing secret — must be a long random string |
| `JWT_EXPIRES_IN` | — | `1h` | JWT expiration duration |
| `PORT` | — | `5000` | Server port |
| `NODE_ENV` | — | `development` | `development` / `production` / `test` |
| `MAX_FAILED_LOGIN_ATTEMPTS` | — | `5` | Failed attempts before temporary account lockout |

Copy `.env.example` to `.env` and fill in values before running. **Never commit `.env` to git.**

---

## 16. Change Log

### v1.0.0 — 2026-09-07
- Initial Module 1 implementation
- `users`, `hospitals`, `audit_logs` schemas created
- JWT authentication implemented (`requireAuth`)
- RBAC implemented (`requireRole`)
- Hospital isolation implemented (`requireHospitalAccess`)
- Admin approval workflow implemented
- Audit logging implemented (`AuditService`)
- Failed login tracking and temporary lockout implemented
- Change-password validation hardened to reject empty password values with `400 VALIDATION_ERROR`
- Complete authentication and authorization test suite implemented
- All 29 Module 1 tests passing
- Module 1 project contracts frozen (roles, identifiers, API paths, error codes, JWT payload, middleware interfaces)

---

## 17. Completion Rule

**Module 1 is considered COMPLETE for implementation when ALL of the following are true:**

- [x] `users`, `hospitals`, `audit_logs` schemas exist and are indexed
- [x] Authentication works (register, login, logout, me, change-password)
- [x] JWT issuance, validation, expiration, and error handling work correctly
- [x] RBAC works (`requireAuth`, `requireRole`, centralized role definitions)
- [x] Hospital isolation works (`requireHospitalAccess` — operator cannot access another hospital)
- [x] Account approval/rejection/suspension workflow works
- [x] Failed-login tracking and temporary lockout work
- [x] Password-change validation works correctly
- [x] Audit logs are created for authentication and authorization events
- [x] Sensitive audit metadata is stripped
- [x] Password hashes are not exposed through normal API responses
- [x] Complete automated test suite passes (29/29)
- [x] API contract is documented (this README, Section 6)
- [x] Integration interfaces are documented (this README, Section 14)

### Module 1 Final Status

**IMPLEMENTATION COMPLETE ✅**

Verified with:

```text
Test Suites: 1 passed, 1 total
Tests:       29 passed, 29 total
```

Additional manual penetration testing, input-fuzzing, and production deployment security review are recommended as broader project-level security hardening activities. They do not block the current Module 1 implementation completion.

> **Module 1 can now be marked complete in the project-wide architecture tracker.**

