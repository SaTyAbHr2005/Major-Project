import { create } from 'zustand';
import authService from '../services/authService';
import socketService from '../services/socketService';
import { useUiStore } from './uiStore';
import { useMLStore } from './mlStore';
import { useFederationStore } from './federationStore';
import { useAdminStore } from './adminStore';
import { useRequestsStore } from './requestsStore';
import { useCommunicationStore } from './communicationStore';

/**
 * Session (Module 1). The JWT itself stays in authService, the only place it is read.
 * Starting or ending a session resets every data store, so nothing one user loaded
 * (datasets, training runs, federation jobs...) is still in memory for the next.
 */
const resetDataStores = () => {
  [useMLStore, useFederationStore, useAdminStore, useRequestsStore, useCommunicationStore]
    .forEach((store) => store.getState().reset());
};

let bootstrapping = null;   // the one session-restore of this page load

export const useAuthStore = create((set) => {
  const startSession = (user) => {
    resetDataStores();
    set({ user, status: 'authenticated' });
    socketService.connect();
    if (user.role === 'hospital_operator') useMLStore.getState().connect(user.hospital_id);
  };

  return {
    user: null,
    status: 'bootstrapping',  // bootstrapping | authenticated | unauthenticated

    /**
     * Restores the session from a stored token on page load. Runs once per page: React StrictMode
     * mounts App twice in development, and a second startSession() would reset the data stores
     * after the screens had already requested their data, leaving them on "Loading..." for good.
     */
    bootstrap: () => (bootstrapping ??= (async () => {
      const user = authService.getToken() ? await authService.getMe() : null;
      if (!user) {
        authService.clearSession();
        set({ user: null, status: 'unauthenticated' });
        return;
      }
      startSession(user);
    })()),

    login: async (identifier, password, role = null, remember = true) => {
      const { user } = await authService.login(identifier, password, role, remember);
      startSession(user);
      useUiStore.getState().setActiveScreen('dashboard');
      return user;
    },

    register: (userData) => authService.register(userData),
    changePassword: (currentPassword, newPassword) => authService.changePassword(currentPassword, newPassword),

    logout: async () => {
      await authService.logout();
      socketService.disconnect();
      resetDataStores();
      set({ user: null, status: 'unauthenticated' });
      useUiStore.getState().setActiveScreen('login');
    },
  };
});

export const selectRole = (s) => s.user?.role ?? null;
export const selectHospitalId = (s) => s.user?.hospital_id ?? null;
/** Hospital name from /auth/me (Module 1); falls back to the id if the hospital record has none. */
export const selectHospitalName = (s) => s.user?.hospital_name || s.user?.hospital_id || '';

const ROLE_LABELS = { admin: 'Consortium admin', researcher: 'Researcher', hospital_operator: 'Hospital operator' };

/** The signed-in user with a few display helpers (initials, role label). */
export const useCurrentUser = () => {
  const user = useAuthStore((s) => s.user);
  const name = user?.name || '';
  return {
    ...user,
    name,
    shortName: name.split(',')[0],
    initials: name.split(/\s+/).filter((w) => /^[A-Za-z]/.test(w) && !/^(dr|prof)\.?$/i.test(w))
      .slice(0, 2).map((w) => w[0].toUpperCase()).join('') || '?',
    roleTag: ROLE_LABELS[user?.role] || '',
  };
};
