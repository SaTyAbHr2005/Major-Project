import React, { createContext, useContext, useState, useEffect } from 'react';
import { authService } from '../services/authService';

const AppContext = createContext();

export const AppProvider = ({ children }) => {
  const [activeScreen, setActiveScreen] = useState('home');
  const [sessionUser, setSessionUser] = useState(() => authService.getCurrentUser());
  const [isLoggedIn, setIsLoggedIn] = useState(() => authService.isAuthenticated());
  const [userRole, setUserRole] = useState(() => {
    const u = authService.getCurrentUser();
    return u?.role || 'researcher';
  });

  const [activeModal, setActiveModal] = useState(null); // 'haltTraining' | 'quarantine' | 'exportWeights' | 'switchRole'
  const [activeMockState, setActiveMockState] = useState('normal'); // 'normal' | 'loading' | 'empty' | 'error'

  const [sessionStatus, setSessionStatus] = useState('running'); // 'running' | 'paused' | 'stopped'
  const [selectedHospitalId, setSelectedHospitalId] = useState('0x9c31be');
  const [selectedExperimentId, setSelectedExperimentId] = useState('EXP-2025-084');
  const [selectedModelId, setSelectedModelId] = useState('effnet-b0');
  
  const [notificationsCount, setNotificationsCount] = useState(3);
  const [messagesCount, setMessagesCount] = useState(2);

  // Synchronize on mount if token exists
  useEffect(() => {
    const existing = authService.getCurrentUser();
    if (existing) {
      setSessionUser(existing);
      setUserRole(existing.role || 'researcher');
      setIsLoggedIn(true);
    }
  }, []);

  const login = async (email, password, role = null, remember = true) => {
    const { user } = await authService.login(email, password, role, remember);
    setSessionUser(user);
    setUserRole(user.role || 'researcher');
    setIsLoggedIn(true);
    setActiveScreen('dashboard');
    return user;
  };

  const register = async (userData) => {
    const newUser = await authService.register(userData);
    return newUser;
  };

  const logout = async () => {
    await authService.logout();
    setSessionUser(null);
    setIsLoggedIn(false);
    setActiveScreen('login');
  };

  // Compose active currentUser representation combining DB user attributes with workstation telemetry
  const currentUser = {
    name: sessionUser?.name || 'Dr. Elena Rostova, M.D., Ph.D.',
    shortName: sessionUser?.name ? sessionUser.name.split(',')[0] : 'Dr. Elena Rostova',
    email: sessionUser?.email || 'e.rostova@med.stanford.edu',
    user_id: sessionUser?.user_id || 'USR_r1e8a9d3c5f2',
    title: sessionUser?.role === 'admin'
      ? 'Consortium Root Security Administrator'
      : 'Lead FL Investigator & Associate Professor of Biomedical Informatics',
    roleTag: (sessionUser?.role || userRole) === 'admin' ? 'Consortium Admin Tier-1' : 'Lead FL Investigator',
    affiliation: sessionUser?.hospital_id
      ? `Hospital Node ${sessionUser.hospital_id}`
      : 'Stanford Medicine AI Lab & Center for Cancer Systems Biology',
    orcid: '0000-0002-8149-9231',
    npi: 'CA-G88421 / NPI #1982736450',
    keyId: sessionUser?.user_id ? `0x${sessionUser.user_id.replace('USR_', '')}...SGX` : '0x9b2d8e41f0c2a88d710e39bc02bfa4e877ca0d1a49f7b11d9a24ec08f237bc90',
    fipsToken: 'YubiKey 5C FIPS (#8839-4029-A) Level 3',
    sessionDuration: '07:42:19',
    enclaveAttestation: 'Intel SGX2 FIPS Validated'
  };

  const privacyMetrics = {
    epsilon: 1.24,
    maxEpsilon: 1.50,
    delta: '1e-5',
    noiseMultiplier: 0.85,
    clipNorm: 1.0,
    percentage: 82.6
  };

  const quorumStatus = {
    connected: 8,
    total: 8,
    statusText: '100% Federated Quorum met',
    fedAvgVersion: 'v3.2',
    lastSync: '12s ago'
  };

  return (
    <AppContext.Provider
      value={{
        activeScreen,
        setActiveScreen,
        userRole,
        setUserRole,
        isLoggedIn,
        setIsLoggedIn,
        sessionUser,
        login,
        register,
        logout,
        activeModal,
        setActiveModal,
        activeMockState,
        setActiveMockState,
        sessionStatus,
        setSessionStatus,
        selectedHospitalId,
        setSelectedHospitalId,
        selectedExperimentId,
        setSelectedExperimentId,
        selectedModelId,
        setSelectedModelId,
        notificationsCount,
        setNotificationsCount,
        messagesCount,
        setMessagesCount,
        // Compatibility aliases for Hospital components
        activeTab: activeScreen,
        setActiveTab: setActiveScreen,
        hospitalId: sessionUser?.hospital_id || 'HOSP_000001',
        hospitalName: sessionUser?.hospital_name || 'St. Jude Clinical Research & AI Node',
        // Hospital ML state (datasets, hardware, training) lives in MLContext.
        notifications: [
          { id: '1', title: 'New FL Training Round Invitation', time: '10m ago', read: false },
          { id: '2', title: 'Global Model Aggregation Sync', time: '1h ago', read: true }
        ],
        messages: [
          { id: '1', sender: 'Consortium Orchestrator', text: 'Training session request approved by Security Council.', time: '10:30 AM' }
        ],
        setMessages: () => {},
        user: sessionUser || currentUser,
        
        currentUser,
        privacyMetrics,
        quorumStatus
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => useContext(AppContext);
